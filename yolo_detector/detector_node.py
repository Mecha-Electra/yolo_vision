import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import message_filters
from geometry_msgs.msg import PoseWithCovariance, Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
from collections import defaultdict
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

class ObjectTriggerNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')

        from ultralytics import YOLO

        self.category_map = {
            # Cleaning supplies
            "Brush": "Cleaning supplies",
            "Cloth": "Cleaning supplies",
            "Oil": "Cleaning supplies",
            "Soap": "Cleaning supplies",
            "Sponge": "Cleaning supplies",
            "Steel wool": "Cleaning supplies",
            # Drinks
            "Gatorade": "Drinks",
            "Guarana": "Drinks",
            "Juice": "Drinks",
            "Nescau": "Drinks",
            "uaizinho cola": "Drinks",
            "uaizinho lemon": "Drinks",
            # Fruits
            "Apple": "Fruits",
            "Green apple": "Fruits",
            "Lemon": "Fruits",
            "Orange": "Fruits",
            "Passion fruit": "Fruits",
            "Pear": "Fruits",
            # Others
            "Bag": "Others",
            #Pantry Items
            "Corn flour": "Pantry Items",
            "Corn": "Pantry Items",
            "Gelatin": "Pantry Items",
            "Olives": "Pantry Items",
            "Pepper": "Pantry Items",
            "Powdered broth": "Pantry Items",
            # Snacks
            "Gummy bear": "Snacks",
            "Peanuts": "Snacks",
            "Torcida pepper": "Snacks",
            "Torcida vinagrette": "Snacks",
            "Tucs": "Snacks",
            "Wavy potato": "Snacks"
        }

        self.declare_parameter('model', 'yolov8n.pt')
        model_name = self.get_parameter('model').get_parameter_value().string_value
        package_share_directory = get_package_share_directory('yolo_detector')
        model_path = os.path.join(package_share_directory, 'resource', model_name)
        self.get_logger().info(f"Carregando o modelo YOLO de: {model_path}")
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_messages = None

        self.objects_publisher = self.create_publisher(Detection3DArray, 'objects/detections', 10)
        self.image_publisher = self.create_publisher(Image, 'objects/annotated_image', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera/color/camera_info', qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 10, 0.1)
        self.ts.registerCallback(self.image_cache_callback)

        self.trigger_sub = self.create_subscription(
            Empty, 'object_detector/trigger', self.trigger_callback, 10)

        self.is_detecting = False
        self.detection_start_time = 0
        self.frame_count = 0
        self.collected_detections = defaultdict(list)
        
        # --- NOVOS ATRIBUTOS PARA O ÚLTIMO FRAME ---
        self.last_color_frame = None
        self.last_frame_detections = [] # Armazena (x1, y1, x2, y2, class_name, confidence) do último frame

        self.get_logger().info('Nó Detector de Objetos Robusto iniciado. Aguardando gatilho em /object_detector/trigger')

    def get_category_for_object(self, object_name):
        return self.category_map.get(object_name, "Categoria Desconhecida")

    def image_cache_callback(self, color_msg, depth_msg, info_msg):
        self.latest_messages = (color_msg, depth_msg, info_msg)
        if self.is_detecting:
            self.process_detection_frame()

    def trigger_callback(self, msg):
        if self.is_detecting:
            self.get_logger().warn("Ciclo de detecção já em andamento. Gatilho ignorado.")
            return
        self.get_logger().info('Gatilho recebido! Iniciando ciclo de detecção de 20 frames...')
        self.is_detecting = True
        self.detection_start_time = time.time()
        self.frame_count = 0
        self.collected_detections.clear()
        self.last_color_frame = None # Resetar para garantir que pegamos o último do ciclo
        self.last_frame_detections = [] # Limpar as detecções do último frame anterior

    def process_detection_frame(self):
        elapsed_time = time.time() - self.detection_start_time
        if elapsed_time > 22 or self.frame_count >= 20:
            self.finalize_detection()
            return
        
        if self.latest_messages is None:
            return

        self.frame_count += 1
        self.get_logger().info(f'Processando frame {self.frame_count}/20...')

        color_msg, depth_msg, info_msg = self.latest_messages
        
        if self.camera_intrinsics is None:
            self.camera_intrinsics = np.array(info_msg.k).reshape(3, 3)

        cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        
        results = self.model(cv_image, verbose=False)

        # --- NOVO: Armazenar o último frame e suas detecções ---
        self.last_color_frame = cv_image.copy() # Copiar para não ser modificado
        current_frame_detections_for_image = [] # Detecções para este frame

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])
                if confidence > 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]

                    box_radius = 5
                    y_start, y_end = max(0, cy - box_radius), min(depth_image.shape[0], cy + box_radius)
                    x_start, x_end = max(0, cx - box_radius), min(depth_image.shape[1], cx + box_radius)
                    depth_roi = depth_image[y_start:y_end, x_start:x_end]
                    non_zero_depths = depth_roi[depth_roi > 0]

                    if non_zero_depths.size > 0:
                        depth_mm = np.median(non_zero_depths)
                        depth_m = depth_mm / 1000.0
                        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
                        cx_cam, cy_cam = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
                        x_cam = (cx - cx_cam) * depth_m / fx
                        y_cam = (cy - cy_cam) * depth_m / fy
                        self.collected_detections[class_name].append((x_cam, y_cam, depth_m, x1, y1, x2, y2))        

                        # Armazenar para o print do último frame
                        current_frame_detections_for_image.append((x1, y1, x2, y2, class_name, confidence))
        
        # O último frame processado terá suas detecções armazenadas aqui
        self.last_frame_detections = current_frame_detections_for_image


    def finalize_detection(self):
        self.get_logger().info("Ciclo de detecção concluído. Calculando médias e publicando...")
        self.is_detecting = False

        detection_array_msg = Detection3DArray()
        detection_array_msg.header.stamp = self.get_clock().now().to_msg()
        detection_array_msg.header.frame_id = 'camera_color_optical_frame'

        # --- NOVO: Dicionário para guardar as deteções médias para desenhar no print final ---
        final_detections_to_draw = []

        for class_name, detections in self.collected_detections.items():
            # Verifique se a coleta de dados foi modificada (agora tem 7 elementos: X, Y, Z, x1, y1, x2, y2)
            if len(detections) >= 4 and len(detections[0]) == 7: 
                all_data = np.array(detections)
                
                # Média das posições 3D (índices 0, 1, 2)
                avg_coords = np.mean(all_data[:, 0:3], axis=0) # [X, Y, Z]
                
                # Média das coordenadas da Bbox (índices 3, 4, 5, 6)
                # Calculamos a média de cada canto separadamente
                avg_bbox = np.mean(all_data[:, 3:7], axis=0).astype(int) # [avg_x1, avg_y1, avg_x2, avg_y2]
                
                # ... (Cálculo da categoria, criação da Pose, etc. - Código existente)
                category = self.get_category_for_object(class_name)

                # ... (Criação e preenchimento de Detection3D e publishing - Código existente)
                self.get_logger().info(f"Objeto confiável detectado! Categoria: '{category}', Objeto: '{class_name}', Posição média: [X: {avg_coords[0]:.2f}, Y: {avg_coords[1]:.2f}, Z: {avg_coords[2]:.2f}]")

                detection = Detection3D()
                detection.header = detection_array_msg.header

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = 1.0 
                
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = avg_coords
                hypothesis.pose.pose = pose
                
                detection.results.append(hypothesis)
                detection_array_msg.detections.append(detection)

                # Adicionar a detecção para desenhar no frame final
                # Usamos a última box detectada para a posição visual, mas a classe e categoria confirmadas
                # O ideal seria ter uma média da box também, mas para fins visuais a última é suficiente.
                # Precisamos encontrar a box original para essa classe específica no `last_frame_detections`
                found_box = next((box_info for box_info in self.last_frame_detections if box_info[4] == class_name), None)
                if found_box:
                    final_detections_to_draw.append({
                        'bbox': found_box[:4], # (x1, y1, x2, y2)
                        'class_name': class_name,
                        'category': category
                    })

                # Adicionar a **detecção média** para desenhar no frame final
                final_detections_to_draw.append({
                    'bbox': tuple(avg_bbox), # (avg_x1, avg_y1, avg_x2, avg_y2)
                    'class_name': class_name,
                    'category': category
                })


        self.objects_publisher.publish(detection_array_msg)
        self.get_logger().info(f"Publicada lista final com {len(detection_array_msg.detections)} objetos confiáveis em /objects/detections")

        # --- NOVO: Lógica para salvar e publicar o print do último frame ---
        if self.last_color_frame is not None:
            annotated_image = self.last_color_frame.copy()
            
            # O loop de desenho já está correto e usará as coordenadas médias
            for det in final_detections_to_draw:
                x1, y1, x2, y2 = det['bbox'] # Aqui são as coordenadas **médias**
                class_name = det['class_name']
                category = det['category']

                # Desenha o retângulo
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde

                # Prepara o texto
                text = f"{class_name} ({category})"
                # Posição do texto (acima do bounding box)
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 # Ajusta a posição para não sair da imagem

                # Adiciona o texto ao frame
                cv2.putText(annotated_image, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # --- SALVAR IMAGEM LOCALMENTE (OPCIONAL, BOM PARA DEBUG) ---
            output_dir = os.path.expanduser('~/ros2_object_detector_prints')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = os.path.join(output_dir, f"detection_snapshot_{timestamp}.png")
            cv2.imwrite(output_filename, annotated_image)
            self.get_logger().info(f"Print do último frame salvo em: {output_filename}")

            # Publica a imagem anotada no tópico do ROS2
            ros_annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            ros_annotated_image_msg.header.stamp = self.get_clock().now().to_msg()
            ros_annotated_image_msg.header.frame_id = 'camera_color_optical_frame' # Adapte se o frame_id for diferente
            self.image_publisher.publish(ros_annotated_image_msg)
            self.get_logger().info("Imagem anotada publicada em /objects/annotated_image")


        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("Processo finalizado. Aguardando novo gatilho...")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTriggerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()