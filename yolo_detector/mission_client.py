import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from yolo_detector.action import DetectCrowd

class MissionClientNode(Node):
    def __init__(self):
        super().__init__('mission_client_node')
        self._action_client = ActionClient(self, DetectCrowd, 'detect_crowd')

    def send_goal(self):
        goal_msg = DetectCrowd.Goal()
        goal_msg.trigger_detection = 1 # Apenas um gatilho para iniciar

        self.get_logger().info('Aguardando pelo servidor de ação...')
        self._action_client.wait_for_server()

        self.get_logger().info('Enviando objetivo para o servidor...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Objetivo rejeitado :(')
            return

        self.get_logger().info('Objetivo aceito :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('--- RESULTADO DA MISSÃO RECEBIDO ---')
        self.get_logger().info(f'Tamanho da multidão: {result.crowd_size}')
        if result.crowd_size > 0:
            coords = result.target_person_coords
            self.get_logger().info(f'Coordenadas do alvo: X={coords.x:.2f}, Y={coords.y:.2f}, Z={coords.z:.2f}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback do servidor: {feedback_msg.feedback.status}')

def main(args=None):
    rclpy.init(args=args)
    client_node = MissionClientNode()
    client_node.send_goal()
    rclpy.spin(client_node)

if __name__ == '__main__':
    main()
