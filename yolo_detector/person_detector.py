import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import message_filters
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_ros
from tf_transformations import quaternion_from_euler
import pyrealsense2 as rs

class ClosestPersonTFNode(Node):
    """
    Nó ROS 2 para detectar a pessoa mais próxima continuamente em relação ao frame da câmera
    e publicar sua transformada TF.
    """
    def __init__(self):
        super().__init__('closest_person_tf_node')
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        
        # Parâmetro de filtro (distância máxima para considerar uma pessoa)
        self.declare_parameter('distance_threshold_m', 3.0)
        self.distance_threshold = self.get_parameter('distance_threshold_m').value

        # Broadcaster TF para publicar a transformada da pessoa mais próxima
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.intrinsics_K = None
        self.width = None
        self.height = None

        # Publisher para o ponto 3D da pessoa alvo (útil para visualização/debug)
        self.target_publisher = self.create_publisher(PointStamped, 'target_person/point', 10)

        # 1. Subscribers para dados da câmera
        self.info_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10)

        self.color_subscriber = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_subscriber = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        # 2. Sincronização aproximada de tempo
        # O callback será chamado sempre que um novo conjunto sincronizado de mensagens for recebido.
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber], 10, 0.1)
        self.time_synchronizer.registerCallback(self.synced_camera_callback)
            
        self.get_logger().info(f"Nó 'closest_person_tf_node' iniciado. Modo contínuo ativado.")
        self.get_logger().info(f"Distância máxima de detecção: {self.distance_threshold:.2f}m.")

    def info_callback(self, msg):
        if self.intrinsics_K is None:
            # Matriz K é (fx, 0, cx, 0, fy, cy, 0, 0, 1) em formato linear
            # msg.k é o formato row-major, que é ideal para o array 3x3
            K = np.array(msg.k).reshape((3, 3))
            
            self.intrinsics_K = K
            self.width = msg.width
            self.height = msg.height
            
            self.get_logger().info('Parâmetros intrínsecos da câmera recebidos!')
            self.destroy_subscription(self.info_subscriber)

    def publish_target_tf(self, coords):
        """Publica a transformada TF da pessoa alvo."""
        
        parent_frame_id = 'camera_color_optical_frame'
        child_frame_id = 'target_person_frame'
        
        # Quatérnion de identidade (sem rotação)
        q = quaternion_from_euler(0, 0, 0)
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame_id
        t.child_frame_id = child_frame_id
        
        # Posição (translação)
        t.transform.translation.x = coords[0]
        t.transform.translation.y = coords[1]
        t.transform.translation.z = coords[2]
        
        # Orientação (rotação)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().debug(f"TF da pessoa mais próxima publicado: {child_frame_id} @ Z={coords[2]:.2f}m")
        
    def project_pixel_to_3d(self, px, py, depth_m):
        """Projeta coordenadas de pixel (cx, cy) para coordenadas 3D (x, y, z) no frame da câmera."""
        if self.intrinsics_K is None:
            self.get_logger().error("Matriz de Intrínsecos da Câmera não inicializada.")
            return None
            
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = self.intrinsics_K[0, 0]
        fy = self.intrinsics_K[1, 1]
        cx_cam = self.intrinsics_K[0, 2]
        cy_cam = self.intrinsics_K[1, 2]
        
        # Fórmula de projeção (inversa)
        # x_cam = (px - cx_cam) * depth_m / fx
        # y_cam = (py - cy_cam) * depth_m / fy
        # z_cam = depth_m
        
        x_cam = (px - cx_cam) * depth_m / fx
        y_cam = (py - cy_cam) * depth_m / fy
        
        return (x_cam, y_cam, depth_m) # O eixo Z é a profundidade

    def get_robust_depth(self, depth_image, cx, cy, radius=5):
        """Calcula a profundidade robusta usando a mediana de uma ROI (em mm)."""
        y_start, y_end = max(0, cy - radius), min(depth_image.shape[0], cy + radius)
        x_start, x_end = max(0, cx - radius), min(depth_image.shape[1], cx + radius)
        depth_roi = depth_image[y_start:y_end, x_start:x_end]
        
        non_zero_depths = depth_roi[depth_roi > 0]

        if non_zero_depths.size > 0:
            return np.median(non_zero_depths)
        else:
            return 0

    def synced_camera_callback(self, color_msg, depth_msg):
        """Callback acionado sempre que um novo conjunto de frames sincronizados é recebido."""

        # Usa o timestamp da mensagem de cor
        if self.intrinsics_K is None: # Linha Correta
            self.get_logger().warn('Aguardando parâmetros intrínsecos da câmera... Imagem ignorada.')
            return

        self.get_logger().info('Formatos convertidos')

        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            #self.get_logger().info('Convertendo formato')
        except Exception as e:
            self.get_logger().error(f'Falha ao converter imagens: {e}')
            return

        # Executa a detecção YOLO (apenas a classe 'person' = 0)
        results = self.model(cv_image, classes=[0], verbose=False) 
        
        detected_people = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence > 0.6: # Filtro de confiança
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    depth_mm = self.get_robust_depth(depth_image, cx, cy)

                    if depth_mm > 0:
                        depth_m = depth_mm / 1000.0
                        
                        # Filtro de distância máxima
                        if depth_m <= self.distance_threshold:
                            coords_3d = self.project_pixel_to_3d(cx, cy, depth_m)
                            if coords_3d:
                                # Apenas o Z é necessário para o min(), mas guardamos tudo
                                detected_people.append({'coords_3d': coords_3d})

        if not detected_people:
            # self.get_logger().debug("Nenhuma pessoa detectada dentro do limite de distância.")
            return

        # Seleciona a pessoa mais próxima (menor coordenada Z)
        closest_person = min(detected_people, key=lambda p: p['coords_3d'][2])
        target_coords = closest_person['coords_3d']
        
        # 1. Publica o TF da pessoa alvo
        self.publish_target_tf(target_coords)

        # 2. Publica o PointStamped
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'camera_color_optical_frame'
        point_msg.point.x, point_msg.point.y, point_msg.point.z = target_coords
        self.target_publisher.publish(point_msg)

        self.get_logger().info(f"Detectado: {len(detected_people)} pessoa(s). Alvo (mais próxima): Z={target_coords[2]:.2f}m")


def main(args=None):
    rclpy.init(args=args)
    node = ClosestPersonTFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()