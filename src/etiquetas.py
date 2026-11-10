"""Etiquetas legibles de los códigos de la Encuesta de Estructura Salarial.

Fuente: diseño de registro del INE (dr_EES_2022.xlsx, hojas "Tablas1" y "Tablas2").
Las descripciones oficiales se han abreviado para que se lean bien en gráficos y en la web;
el código original se conserva siempre en el dataset.
"""

import pandas as pd

ETIQUETAS = {
    "NUTS1": {  # Región (NUTS 1)
        "1": "Noroeste",
        "2": "Noreste",
        "3": "Comunidad de Madrid",
        "4": "Centro",
        "5": "Este",
        "6": "Sur",
        "7": "Canarias",
    },
    "CNACE": {  # Sector de actividad (agrupación CNAE-09)
        "B0": "Industrias extractivas",
        "C1": "Alimentación, bebidas, textil y calzado",
        "C2": "Madera, corcho y papel",
        "C3": "Artes gráficas",
        "C4": "Química, farmacia, refino y plásticos",
        "C5": "Otros productos minerales no metálicos",
        "C6": "Metalurgia y productos metálicos",
        "C7": "Informática, electrónica y maquinaria",
        "C8": "Vehículos, material de transporte y muebles",
        "D0": "Energía eléctrica y gas",
        "E0": "Agua, saneamiento y residuos",
        "F0": "Construcción",
        "G1": "Comercio mayorista y venta de vehículos",
        "G2": "Comercio minorista",
        "H1": "Transporte de viajeros y mercancías",
        "H2": "Almacenamiento y correos",
        "I0": "Hostelería",
        "J0": "Información y comunicaciones",
        "K0": "Finanzas y seguros",
        "L0": "Actividades inmobiliarias",
        "M0": "Actividades profesionales, científicas y técnicas",
        "N0": "Actividades administrativas y servicios auxiliares",
        "O0": "Administración pública y defensa",
        "P0": "Educación",
        "Q0": "Sanidad y servicios sociales",
        "R0": "Actividades artísticas y recreativas",
        "S0": "Otros servicios",
    },
    "ESTRATO2": {  # Tamaño del centro (0 y 4 son agrupaciones por anonimización)
        "0": "Sin desagregar (todos los tamaños)",
        "1": "De 1 a 49 trabajadores",
        "2": "De 50 a 199 trabajadores",
        "3": "200 o más trabajadores",
        "4": "50 o más (sin desagregar)",
    },
    "CONTROL": {"1": "Público", "2": "Privado"},
    "MERCADO": {
        "1": "Local o regional",
        "2": "Nacional",
        "3": "Unión Europea",
        "4": "Mundial",
    },
    "REGULACION": {  # Tipo de convenio
        "1": "Convenio sectorial estatal",
        "2": "Convenio sectorial de ámbito inferior",
        "3": "Convenio de empresa o grupo",
        "4": "Convenio de centro de trabajo",
        "5": "Otra forma de regulación",
    },
    "SEXO": {"1": "Hombre", "6": "Mujer"},
    "TIPOPAIS": {"1": "España", "2": "Resto del mundo"},
    "CNO1": {  # Ocupación (grandes grupos CNO-11)
        "A0": "Directores y gerentes",
        "B0": "Profesionales de la salud y la enseñanza",
        "C0": "Otros técnicos y profesionales científicos",
        "D0": "Técnicos y profesionales de apoyo",
        "E0": "Empleados de oficina sin atención al público",
        "F0": "Empleados de oficina con atención al público",
        "G0": "Trabajadores de restauración y comercio",
        "H0": "Trabajadores de salud y cuidado de personas",
        "I0": "Trabajadores de protección y seguridad",
        "J0": "Trabajadores cualificados agrícolas y pesqueros",
        "K0": "Trabajadores cualificados de la construcción",
        "L0": "Trabajadores cualificados de la industria",
        "M0": "Operadores de instalaciones y montadores",
        "N0": "Conductores y operadores de maquinaria móvil",
        "O0": "Trabajadores no cualificados en servicios",
        "P0": "Peones de agricultura, industria, construcción y transporte",
        "Q0": "Ocupaciones militares",
    },
    "RESPONSA": {"1": "Con responsabilidad", "0": "Sin responsabilidad"},
    "ESTU": {  # Nivel de estudios
        "1": "Menos que primaria",
        "2": "Primaria",
        "3": "Secundaria 1ª etapa (ESO)",
        "4": "Secundaria 2ª etapa (Bachillerato, FP media)",
        "5": "FP superior y similares",
        "6": "Diplomatura universitaria y similares",
        "7": "Licenciatura, grado, máster o doctorado",
    },
    "TIPOJOR": {"1": "Tiempo completo", "2": "Tiempo parcial"},
    "TIPOCON": {"1": "Indefinido", "2": "Temporal"},
    "ANOS2": {  # Tramo de edad
        "01": "Menos de 20",
        "02": "20 a 29",
        "03": "30 a 39",
        "04": "40 a 49",
        "05": "50 a 59",
        "06": "60 o más",
    },
}

# Nombres legibles de las columnas etiquetadas
NOMBRES = {
    "NUTS1": "region",
    "CNACE": "sector",
    "ESTRATO2": "tamano_empresa",
    "CONTROL": "control",
    "MERCADO": "mercado",
    "REGULACION": "convenio",
    "SEXO": "sexo",
    "TIPOPAIS": "nacionalidad",
    "CNO1": "ocupacion",
    "RESPONSA": "responsabilidad",
    "ESTU": "estudios",
    "TIPOJOR": "jornada",
    "TIPOCON": "contrato",
    "ANOS2": "edad",
}


def aplicar_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    """Añade una columna categórica legible por cada variable codificada.

    El orden de las categorías sigue el de los códigos del INE, de modo que
    estudios, edad o tamaño de empresa quedan ordenados de forma natural.
    """
    df = df.copy()
    for codigo, nombre in NOMBRES.items():
        mapa = ETIQUETAS[codigo]
        sin_etiqueta = set(df[codigo].unique()) - set(mapa)
        if sin_etiqueta:
            raise ValueError(f"{codigo}: códigos sin etiqueta {sorted(sin_etiqueta)}")
        df[nombre] = pd.Categorical(df[codigo].map(mapa), categories=list(mapa.values()), ordered=False)
    return df
