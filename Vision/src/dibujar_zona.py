import cv2
import numpy as np
import json
import os


def _boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _find_label_pos(x, y, text, font_scale, thickness, placed_boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 4

    candidates = [
        (x + 8,      y - 8),
        (x - w - 8,  y - 8),
        (x + 8,      y + h + 8),
        (x - w - 8,  y + h + 8),
        (x + 8,      y - h - 16),
        (x - w - 8,  y - h - 16),
        (x + 8,      y + h + 24),
        (x - w - 8,  y + h + 24),
    ]

    for tx, ty in candidates:
        box = (tx - margin, ty - h - margin, tx + w + margin, ty + baseline + margin)
        if not any(_boxes_overlap(box, pb) for pb in placed_boxes):
            placed_boxes.append(box)
            return tx, ty

    # fallback: stack below all placed boxes
    max_y = max((pb[3] for pb in placed_boxes), default=y)
    tx, ty = x + 8, max_y + h + 8
    placed_boxes.append((tx - margin, ty - h - margin, tx + w + margin, ty + baseline + margin))
    return tx, ty


def _put_text(img, text, x, y, font_scale, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)


def dibujar_poligono(imagen, puntos, color, opacidad=0.4):
    pts = np.array(puntos, np.int32).reshape((-1, 1, 2))
    overlay = imagen.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
    return cv2.addWeighted(overlay, opacidad, imagen, 1 - opacidad, 0)


def dibujar_nombre_zona(imagen, puntos, nombre, placed_boxes):
    pts = np.array(puntos, np.int32).reshape((-1, 1, 2))
    M = cv2.moments(pts)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = pts[0][0]
    tx, ty = _find_label_pos(cx, cy, nombre, 0.7, 2, placed_boxes)
    _put_text(imagen, nombre, tx, ty, 0.7, (255, 255, 255))


def dibujar_vertices(imagen, puntos, color, placed_boxes):
    for punto in puntos:
        x, y = punto
        coord_text = f"({x},{y})"
        cv2.circle(imagen, (x, y), 5, color, -1)
        tx, ty = _find_label_pos(x, y, coord_text, 0.5, 1, placed_boxes)
        _put_text(imagen, coord_text, tx, ty, 0.5, (255, 255, 255))


def dibujar_ejes(imagen):
    pad = 30
    largo = 120
    color_x = (0, 0, 255)    # rojo
    color_y = (0, 255, 0)    # verde
    color_origen = (255, 255, 255)

    ox, oy = pad, pad

    # X axis arrow (points right)
    cv2.arrowedLine(imagen, (ox, oy), (ox + largo, oy), color_x, 2, tipLength=0.15)
    # Y axis arrow (points down)
    cv2.arrowedLine(imagen, (ox, oy), (ox, oy + largo), color_y, 2, tipLength=0.15)

    # Origin dot and label
    cv2.circle(imagen, (ox, oy), 5, color_origen, -1)
    _put_text(imagen, "(0,0)", ox + 6, oy - 6, 0.5, color_origen)

    # Axis labels
    _put_text(imagen, "X+", ox + largo + 6, oy + 5, 0.5, color_x)
    _put_text(imagen, "Y+", ox - 4, oy + largo + 18, 0.5, color_y)


def procesar_carpeta(carpeta_input, ruta_json, carpeta_output):
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    colores_disponibles = [
        (255, 0, 0),   # Azul
        (0, 255, 0),   # Verde
        (0, 0, 255),   # Rojo
        (0, 255, 255), # Amarillo
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cian
        (255, 128, 0), # Naranja
        (128, 0, 255)  # Violeta
    ]

    with open(ruta_json, 'r', encoding='utf-8') as f:
        configuracion = json.load(f)

    for camara in configuracion:
        nombre_original = camara['camera_name']
        nombre_formateado = nombre_original.replace(" ", "_")

        archivo_encontrado = None
        for f in os.listdir(carpeta_input):
            if os.path.splitext(f)[0] == nombre_formateado:
                archivo_encontrado = f
                break

        if archivo_encontrado:
            ruta_img = os.path.join(carpeta_input, archivo_encontrado)
            imagen = cv2.imread(ruta_img)

            if imagen is None:
                print(f"Error: No se pudo leer la imagen {archivo_encontrado}")
                continue

            img_resultado = imagen.copy()
            placed_boxes = []

            # Pass 1: polygons (no text)
            for i, zona in enumerate(camara['zones']):
                color_zona = colores_disponibles[i % len(colores_disponibles)]
                img_resultado = dibujar_poligono(img_resultado, zona['vertices'], color_zona, opacidad=0.5)

            # Pass 2: zone names (collision-aware)
            for zona in camara['zones']:
                print(f"Dibujando en {archivo_encontrado}: {zona['name']}")
                dibujar_nombre_zona(img_resultado, zona['vertices'], zona['name'], placed_boxes)

            # Pass 3: vertex coordinates (collision-aware, always on top)
            for i, zona in enumerate(camara['zones']):
                color_zona = colores_disponibles[i % len(colores_disponibles)]
                dibujar_vertices(img_resultado, zona['vertices'], color_zona, placed_boxes)

            # Pass 4: axis indicator (drawn last so nothing covers it)
            dibujar_ejes(img_resultado)

            ruta_guardado = os.path.join(carpeta_output, f"COLOREADA_{archivo_encontrado}")
            cv2.imwrite(ruta_guardado, img_resultado)
            print(f"Éxito: {ruta_guardado}")
        else:
            print(f"Aviso: No se encontró la imagen '{nombre_formateado}'")


def procesar_carpeta_fup(carpeta_input, ruta_json, nick_cam, carpeta_output):
    """
    Processes all images in *carpeta_input* and draws flux-counter zones
    defined in the FUP-style JSON (flux_counters / zone_in / zone_middle / zone_out).

    Args:
        carpeta_input  : folder that contains the source images.
        ruta_json      : path to the FUP JSON config file.
        nick_cam       : value of 'nick_cam' in the JSON to use.
        carpeta_output : folder where annotated images will be saved.
    """
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    colores_disponibles = [
        (255, 0, 0),    # Azul  – zone_in
        (0, 255, 0),    # Verde – zone_middle
        (0, 0, 255),    # Rojo  – zone_out
        (0, 255, 255),  # Amarillo
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cian
        (255, 128, 0),  # Naranja
        (128, 0, 255),  # Violeta
    ]

    with open(ruta_json, 'r', encoding='utf-8') as f:
        configuracion = json.load(f)

    # Find the camera entry that matches nick_cam
    camara = next((c for c in configuracion if c.get('nick_cam') == nick_cam), None)
    if camara is None:
        print(f"Error: nick_cam '{nick_cam}' not found in {ruta_json}")
        return

    # Build a flat list of zones from all flux_counters
    zone_keys = ['zone_in', 'zone_middle', 'zone_out']
    zonas = []
    for fc in camara.get('flux_counters', []):
        for key in zone_keys:
            if key in fc:
                zonas.append(fc[key])
        # Also include any extra keys that look like zone dicts
        for k, v in fc.items():
            if k not in ('name', 'interest_point') + tuple(zone_keys):
                if isinstance(v, dict) and 'vertices' in v:
                    zonas.append(v)

    if not zonas:
        print(f"Aviso: no zones found for nick_cam '{nick_cam}'")
        return

    # Process every image in the input folder
    ext_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    archivos = [f for f in os.listdir(carpeta_input)
                if os.path.splitext(f)[1].lower() in ext_validas]

    if not archivos:
        print(f"Aviso: no images found in {carpeta_input}")
        return

    for archivo in archivos:
        ruta_img = os.path.join(carpeta_input, archivo)
        imagen = cv2.imread(ruta_img)
        if imagen is None:
            print(f"Error: could not read {archivo}")
            continue

        img_resultado = imagen.copy()
        placed_boxes = []

        # Pass 1: polygons
        for i, zona in enumerate(zonas):
            color = colores_disponibles[i % len(colores_disponibles)]
            img_resultado = dibujar_poligono(img_resultado, zona['vertices'], color, opacidad=0.5)

        # Pass 2: zone names
        for zona in zonas:
            dibujar_nombre_zona(img_resultado, zona['vertices'], zona['name'], placed_boxes)

        # Pass 3: vertex coordinates
        for i, zona in enumerate(zonas):
            color = colores_disponibles[i % len(colores_disponibles)]
            dibujar_vertices(img_resultado, zona['vertices'], color, placed_boxes)

        # Pass 4: axis indicator
        dibujar_ejes(img_resultado)

        ruta_guardado = os.path.join(carpeta_output, f"COLOREADA_{archivo}")
        cv2.imwrite(ruta_guardado, img_resultado)
        print(f"OK: {ruta_guardado}")


def procesar_imagen_fup(ruta_imagen, ruta_json, nick_cam, carpeta_output):
    """
    Draws flux-counter zones on a single image using a FUP-style JSON config.

    Args:
        ruta_imagen    : path to the source image file.
        ruta_json      : path to the FUP JSON config file.
        nick_cam       : value of 'nick_cam' in the JSON to use.
        carpeta_output : folder where the annotated image will be saved.
    """
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    colores_disponibles = [
        (255, 0, 0),    # Azul  – zone_in
        (0, 255, 0),    # Verde – zone_middle
        (0, 0, 255),    # Rojo  – zone_out
        (0, 255, 255),  # Amarillo
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cian
        (255, 128, 0),  # Naranja
        (128, 0, 255),  # Violeta
    ]

    with open(ruta_json, 'r', encoding='utf-8') as f:
        configuracion = json.load(f)

    camara = next((c for c in configuracion if c.get('nick_cam') == nick_cam), None)
    if camara is None:
        print(f"Error: nick_cam '{nick_cam}' not found in {ruta_json}")
        return

    zone_keys = ['zone_in', 'zone_middle', 'zone_out']
    zonas = []
    for fc in camara.get('flux_counters', []):
        for key in zone_keys:
            if key in fc:
                zonas.append(fc[key])
        for k, v in fc.items():
            if k not in ('name', 'interest_point') + tuple(zone_keys):
                if isinstance(v, dict) and 'vertices' in v:
                    zonas.append(v)

    if not zonas:
        print(f"Aviso: no zones found for nick_cam '{nick_cam}'")
        return

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: could not read {ruta_imagen}")
        return

    img_resultado = imagen.copy()
    placed_boxes = []

    for i, zona in enumerate(zonas):
        color = colores_disponibles[i % len(colores_disponibles)]
        img_resultado = dibujar_poligono(img_resultado, zona['vertices'], color, opacidad=0.5)

    for zona in zonas:
        dibujar_nombre_zona(img_resultado, zona['vertices'], zona['name'], placed_boxes)

    for i, zona in enumerate(zonas):
        color = colores_disponibles[i % len(colores_disponibles)]
        dibujar_vertices(img_resultado, zona['vertices'], color, placed_boxes)

    dibujar_ejes(img_resultado)

    nombre_archivo = os.path.basename(ruta_imagen)
    ruta_guardado = os.path.join(carpeta_output, f"COLOREADA_{nombre_archivo}")
    cv2.imwrite(ruta_guardado, img_resultado)
    print(f"OK: {ruta_guardado}")


def procesar_imagen(ruta_imagen, ruta_json, carpeta_output, indice_camara=0):
    """
    Draws standard zones on a single image using a vertices-style JSON config.

    Args:
        ruta_imagen    : path to the source image file.
        ruta_json      : path to the JSON config file (camera_name / zones format).
        carpeta_output : folder where the annotated image will be saved.
        indice_camara  : index into the JSON array to use (default 0).
    """
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    colores_disponibles = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
    ]

    with open(ruta_json, 'r', encoding='utf-8') as f:
        configuracion = json.load(f)

    camara = configuracion[indice_camara]
    zonas = camara.get('zones', [])

    if not zonas:
        print(f"Aviso: no zones found in {ruta_json} (indice_camara={indice_camara})")
        return

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: could not read {ruta_imagen}")
        return

    img_resultado = imagen.copy()
    placed_boxes = []

    for i, zona in enumerate(zonas):
        color = colores_disponibles[i % len(colores_disponibles)]
        img_resultado = dibujar_poligono(img_resultado, zona['vertices'], color, opacidad=0.5)

    for zona in zonas:
        dibujar_nombre_zona(img_resultado, zona['vertices'], zona['name'], placed_boxes)

    for i, zona in enumerate(zonas):
        color = colores_disponibles[i % len(colores_disponibles)]
        dibujar_vertices(img_resultado, zona['vertices'], color, placed_boxes)

    dibujar_ejes(img_resultado)

    nombre_archivo = os.path.basename(ruta_imagen)
    ruta_guardado = os.path.join(carpeta_output, f"COLOREADA_{nombre_archivo}")
    cv2.imwrite(ruta_guardado, img_resultado)
    print(f"OK: {ruta_guardado}")


def procesar_imagen_labelme(ruta_imagen, ruta_json, carpeta_output):
    """
    Draws LabelMe polygon shapes on a single image using its sidecar JSON.

    Args:
        ruta_imagen    : path to the source image file.
        ruta_json      : path to the LabelMe JSON file (shapes[].label / shapes[].points).
        carpeta_output : folder where the annotated image will be saved.
    """
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    colores_disponibles = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
    ]

    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    zonas = [
        {'name': s['label'], 'vertices': [[int(p[0]), int(p[1])] for p in s['points']]}
        for s in data.get('shapes', [])
        if s.get('shape_type') == 'polygon' and s.get('points')
    ]

    if not zonas:
        print(f"Aviso: no polygon shapes found in {ruta_json}")
        return

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: could not read {ruta_imagen}")
        return

    img_resultado = imagen.copy()
    placed_boxes = []

    for i, zona in enumerate(zonas):
        color = colores_disponibles[i % len(colores_disponibles)]
        img_resultado = dibujar_poligono(img_resultado, zona['vertices'], color, opacidad=0.5)

    for zona in zonas:
        print(f"Dibujando: {zona['name']}")
        dibujar_nombre_zona(img_resultado, zona['vertices'], zona['name'], placed_boxes)

    for i, zona in enumerate(zonas):
        color = colores_disponibles[i % len(colores_disponibles)]
        dibujar_vertices(img_resultado, zona['vertices'], color, placed_boxes)

    dibujar_ejes(img_resultado)

    nombre_archivo = os.path.basename(ruta_imagen)
    ruta_guardado = os.path.join(carpeta_output, f"COLOREADA_{nombre_archivo}")
    cv2.imwrite(ruta_guardado, img_resultado)
    print(f"OK: {ruta_guardado}")


def procesar_carpeta_labelme(carpeta_input, carpeta_output):
    """
    Processes image+JSON pairs in LabelMe format (shapes[].label / shapes[].points).
    Each image must have a matching .json sidecar in the same folder.

    Args:
        carpeta_input  : folder containing the .jpg/.png and .json files.
        carpeta_output : folder where annotated images will be saved.
    """
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    colores_disponibles = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
    ]

    ext_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    archivos = [f for f in os.listdir(carpeta_input)
                if os.path.splitext(f)[1].lower() in ext_validas]

    if not archivos:
        print(f"Aviso: no images found in {carpeta_input}")
        return

    for archivo in archivos:
        nombre_base = os.path.splitext(archivo)[0]
        ruta_json = os.path.join(carpeta_input, nombre_base + '.json')

        if not os.path.exists(ruta_json):
            print(f"Aviso: no JSON sidecar for {archivo}, skipping")
            continue

        with open(ruta_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        zonas = [
            {'name': s['label'], 'vertices': [[int(p[0]), int(p[1])] for p in s['points']]}
            for s in data.get('shapes', [])
            if s.get('shape_type') == 'polygon' and s.get('points')
        ]

        if not zonas:
            print(f"Aviso: no polygon shapes found in {ruta_json}")
            continue

        ruta_img = os.path.join(carpeta_input, archivo)
        imagen = cv2.imread(ruta_img)
        if imagen is None:
            print(f"Error: could not read {archivo}")
            continue

        img_resultado = imagen.copy()
        placed_boxes = []

        for i, zona in enumerate(zonas):
            color = colores_disponibles[i % len(colores_disponibles)]
            img_resultado = dibujar_poligono(img_resultado, zona['vertices'], color, opacidad=0.5)

        for zona in zonas:
            print(f"Dibujando en {archivo}: {zona['name']}")
            dibujar_nombre_zona(img_resultado, zona['vertices'], zona['name'], placed_boxes)

        for i, zona in enumerate(zonas):
            color = colores_disponibles[i % len(colores_disponibles)]
            dibujar_vertices(img_resultado, zona['vertices'], color, placed_boxes)

        dibujar_ejes(img_resultado)

        ruta_guardado = os.path.join(carpeta_output, f"COLOREADA_{archivo}")
        cv2.imwrite(ruta_guardado, img_resultado)
        print(f"OK: {ruta_guardado}")


# --- Configuración de rutas ---
ruta_imagenes = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/DCP/cam_capture"
ruta_json = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/jsons/DCP/vertices.json"
ruta_destino = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/DCP/zonas"

procesar_carpeta(ruta_imagenes, ruta_json, ruta_destino)

# --- DCP – bodega_j (door vertices) ---
procesar_imagen(
    ruta_imagen    = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/DCP/cam_capture/192_168_4_123_2.jpg",
    ruta_json      = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/jsons/DCP/door_vertices.json",
    carpeta_output = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/DCP/zonas",
)

# --- FUP Mall – easton3 (single reference image) ---
procesar_imagen_fup(
    ruta_imagen    = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/easton3.jpg",
    ruta_json      = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/jsons/FUP/easton3.json",
    nick_cam       = "easton3",
    carpeta_output = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/zonas",
)

# --- FUP Mall – easton4 (single reference image) ---
procesar_imagen_fup(
    ruta_imagen    = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/easton4.jpg",
    ruta_json      = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/jsons/FUP/easton4.json",
    nick_cam       = "easton4",
    carpeta_output = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/zonas",
)

# --- Dipreca – captures (LabelMe format) ---
procesar_carpeta_labelme(
    carpeta_input  = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/Dipreca/captures",
    carpeta_output = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/Dipreca/zonas",
)

# --- FUP Mall – easton3 (LabelMe sidecar JSON) ---
procesar_imagen_labelme(
    ruta_imagen    = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/easton3.jpg",
    ruta_json      = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/easton3.json",
    carpeta_output = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/zonas",
)
