import os


def mostrar_estructura(ruta, nivel=0):
    try:
        for elemento in os.listdir(ruta):
            ruta_completa = os.path.join(ruta, elemento)
            print("   " * nivel + "📁 " + elemento if os.path.isdir(ruta_completa) else "   " * nivel + "📄 " + elemento)

            if os.path.isdir(ruta_completa):
                mostrar_estructura(ruta_completa, nivel + 1)
    except PermissionError:
        print("   " * nivel + "⛔ [Acceso denegado]")


# Ruta de la carpeta a inspeccionar
ruta_carpeta = r"C:\Users\Admin\PycharmProjects\IA_Finance_Society_UCN\Archivos Python y Data"  # 🔹 Cambia esta ruta
mostrar_estructura(ruta_carpeta)
