import socket

IP = "127.0.0.1"
PORT = 20100


def send_generic_udp_message(message, consolePrint=False):
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Send the message
        sock.sendto(message.encode(), (IP, PORT))
        if consolePrint:
            print("UDP message sent successfully")
    except socket.error as e:
        print(f"Error sending UDP message: {e}")
    finally:
        # Close the socket
        sock.close()
