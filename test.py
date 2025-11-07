import socket
import urllib.request

def check_internet_socket(host="8.8.8.8", port=53, timeout=3):
    """
    Checks internet connectivity by trying to connect to Google's DNS (8.8.8.8).
    Works even if HTTP is blocked.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception as e:
        print(f"[Socket Test] No connection: {e}")
        return False


def check_internet_http(url="https://www.google.com", timeout=5):
    """
    Checks internet connectivity by trying to fetch a webpage.
    """
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception as e:
        print(f"[HTTP Test] Cannot reach {url}: {e}")
        return False


if __name__ == "__main__":
    print("Testing internet connection...\n")

    socket_ok = check_internet_socket()
    http_ok = check_internet_http()

    if socket_ok and http_ok:
        print("\n✅ Internet connection is working.")
    elif socket_ok and not http_ok:
        print("\n⚠️ Network reachable but HTTP requests are blocked (possible proxy/firewall).")
    else:
        print("\n❌ No internet connection detected.")
import socket
import urllib.request

def check_internet_socket(host="8.8.8.8", port=53, timeout=3):
    """
    Checks internet connectivity by trying to connect to Google's DNS (8.8.8.8).
    Works even if HTTP is blocked.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception as e:
        print(f"[Socket Test] No connection: {e}")
        return False


def check_internet_http(url="https://www.google.com", timeout=5):
    """
    Checks internet connectivity by trying to fetch a webpage.
    """
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception as e:
        print(f"[HTTP Test] Cannot reach {url}: {e}")
        return False


if __name__ == "__main__":
    print("Testing internet connection...\n")

    socket_ok = check_internet_socket()
    http_ok = check_internet_http()

    if socket_ok and http_ok:
        print("\n✅ Internet connection is working.")
    elif socket_ok and not http_ok:
        print("\n⚠️ Network reachable but HTTP requests are blocked (possible proxy/firewall).")
    else:
        print("\n❌ No internet connection detected.")

