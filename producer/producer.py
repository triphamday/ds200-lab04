import socket, json, time
from producer.custom_dataset import scan_images

def send_metadata(root, host='localhost', port=9999, sleep_time=0.01):
    s = socket.socket()
    s.bind((host, port))
    s.listen(1)
    print(f'Producer listening on {host}:{port}...')
    conn, addr = s.accept()
    print('Connected by', addr)
    for rec in scan_images(root):
        msg = json.dumps(rec) + '\n'
        conn.sendall(msg.encode('utf-8'))
        time.sleep(sleep_time)
    conn.close()
    s.close()
    print('Done sending all data.')

if __name__ == "__main__":
    # Dữ liệu train, hoặc val/test tuỳ bạn muốn stream gì
    DATASET_DIR = r"D:\ds200-lab04\dataset\data_split\train"
    send_metadata(DATASET_DIR)
