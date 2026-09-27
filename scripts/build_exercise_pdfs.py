"""Apuntes sencillos, en blanco y negro, con el código enviado al editor."""

from pathlib import Path
from html import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.enums import TA_JUSTIFY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

ACTIVITIES = [
    ("M2_U2_Validacion_y_resiliencia", "Módulo 2 - Unidad 2", "Validación estructurada y resiliencia",
     "m2u2_resiliencia.py", "89%",
     "La actividad consiste en convertir una cadena que devuelve texto libre en una extracción con contrato. Se define EntityExtraction con un tema, una lista de entidades y un sentimiento entre cero y uno. La cadena debe ejecutarse de forma asíncrona y manejar fallos sin reintentar indefinidamente.",
     "Pydantic valida que la respuesta tenga los campos y tipos previstos. Las restricciones agregan reglas de significado, como impedir que un puntaje supere uno. Esto no asegura que el modelo diga la verdad, pero evita que una respuesta mal formada pase como un objeto válido.",
     "El prompt se conecta con el modelo mediante LCEL. with_structured_output pide el contrato y with_retry limita a tres los intentos ante errores transitorios o de formato. El timeout limita la espera total. Una clave inválida no se reintenta, porque es un error permanente. Si se agotan los intentos, esta versión devuelve None de forma controlada; no inventa una extracción.",
     "Se probó localmente la composición asíncrona con un modelo de prueba explícito y el rechazo de una entrada vacía. No se ejecutó una llamada real de OpenAI con este ejercicio. Ticher aprobó el código con 89%. Su principal mejora sugerida es añadir un fallback estratégico; la versión adjunta es exactamente la enviada."),
    ("M5_U3_ReAct_y_memoria", "Módulo 5 - Unidad 3", "Ciclos ReAct y memoria del agente",
     "m5u3_react_memoria.py", "100%",
     "Se construye un agente climatológico con un nodo de modelo y un nodo de herramientas. La primera pregunta menciona Buenos Aires. La segunda pregunta se refiere a la misma ciudad sin volver a nombrarla. El historial se recupera usando el mismo thread_id.",
     "ReAct permite alternar entre elegir una herramienta, ejecutarla y observar su resultado. ToolNode ejecuta la herramienta y devuelve el mensaje correspondiente. tools_condition decide si hay que volver a herramientas o finalizar. add_messages incorpora los nuevos mensajes sin borrar los anteriores.",
     "MemorySaver guarda checkpoints en memoria durante la vida del proceso. El mismo thread_id permite continuar la conversación, pero un reinicio del programa pierde ese almacenamiento. La herramienta get_weather devuelve datos simulados, como en la plantilla de la consigna. No es un pronóstico meteorológico real.",
     "Se ejecutó el grafo real con un modelo de prueba explícito. Se comprobaron dos llamadas a la herramienta para Buenos Aires, ocho mensajes conservados y una segunda sesión aislada para Córdoba. La conexión real con OpenAI está pendiente de una clave válida. Ticher aprobó la implementación con 100%."),
    ("M6_U2_Supervisor", "Módulo 6 - Unidad 2", "Supervisor y delegación dinámica",
     "m6u2_supervisor.py", "94%",
     "El supervisor decide si el siguiente paso corresponde al Analista, al Escritor o a FINALIZAR. Lee el historial y coordina el trabajo sin realizar él mismo las tareas de los especialistas. Se conserva el ejemplo de especialistas con datos fijos suministrado por la consigna.",
     "El grafo comienza en el supervisor. Una arista condicional dirige el flujo según next. Ambos especialistas vuelven al supervisor para revisar el avance. El estado contiene messages con add_messages, la siguiente ruta y un contador de decisiones. RoutingDecision restringe la respuesta del modelo a las tres opciones permitidas.",
     "Una cadena de agentes puede entrar en un ciclo que consuma tiempo y dinero. Esta solución limita las delegaciones, configura recursion_limit y aplica un timeout al modelo. Los especialistas de esta actividad devuelven textos fijos de demostración; no son un análisis de datos de una empresa. En el proyecto integrado los especialistas trabajan con artefactos y evidencia.",
     "La prueba local ejecutó LangGraph con un router de prueba. Se verificó la secuencia Analista, Escritor y FINALIZAR, la conservación del historial y tres ejecuciones concurrentes con estados separados. No se presenta esa prueba como una llamada real a un modelo. Ticher aprobó el código con 94%."),
    ("M7_U2_FastAPI_y_Redis", "Módulo 7 - Unidad 2", "Procesamiento asíncrono con FastAPI y Redis",
     "m7u2_fastapi_redis.py", "95%",
     "La API recibe una consulta, crea un identificador y devuelve 202 sin esperar a que termine el trabajo. Redis conserva el estado y una cola de identificadores. El cliente consulta /status/{job_id} hasta que el trabajo esté completed o failed.",
     "Asincronía no significa que una tarea costosa desaparezca. Significa que el servidor puede aceptar otras solicitudes mientras el worker espera. La transacción de Redis registra primero pending y luego encola el identificador, evitando que un worker rápido encuentre una tarea sin estado.",
     "Tres workers consumen con blpop. El procesamiento usa asyncio.sleep porque la consigna pide una simulación. Se registran running, completed o failed y se limita el tiempo de ejecución. La validación Pydantic rechaza consultas vacías. El polling distingue un trabajo inexistente, 404, de una falla de Redis, 503. El estado expira a las veinticuatro horas.",
     "Se probó con Redis real y cinco solicitudes concurrentes. Las cinco recibieron 202 y terminaron completed, sin perder sus identificadores. También se verificaron 422 para entrada inválida y 404 para un identificador inexistente. Esta cola simple no recupera un trabajo running tras una caída abrupta; el proyecto final incorpora checkpoints y recuperación. Ticher aprobó el código con 95%."),
]

body = ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=10)
heading = ParagraphStyle("Heading", fontName="Times-Bold", fontSize=14, leading=18,
                         spaceBefore=12, spaceAfter=10)
title = ParagraphStyle("Title", fontName="Times-Bold", fontSize=18, leading=22, spaceAfter=14)
code_style = ParagraphStyle("Code", fontName="Courier", fontSize=7.5, leading=10,
                           spaceAfter=6)


def page_number(canvas, doc):
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(A4[0] / 2, 30, str(doc.page))


for slug, module, topic, filename, grade, task, concept, solution, verification in ACTIVITIES:
    story = [Paragraph(escape(module), heading), Paragraph(escape(topic), title),
             Paragraph("AI Engineering<br/>Mauro Tomas Proto Cassina<br/>26 de septiembre de 2026", body),
             Spacer(1, 12), Paragraph("Índice", heading),
             Paragraph("1. Consigna<br/>2. Conceptos<br/>3. Resolución<br/>4. Verificación<br/>5. Código enviado", body)]
    for name, content in [("1. Consigna", task), ("2. Conceptos", concept),
                          ("3. Resolución", solution), ("4. Verificación", verification)]:
        story.extend([Paragraph(name, heading), Paragraph(escape(content), body)])
    story.extend([PageBreak(), Paragraph("5. Código enviado", heading),
                  Paragraph(escape(f"Archivo acompañante: {filename}. Las líneas largas se ajustan solo para la lectura del PDF; el archivo Python conserva el código original."), body)])
    source = (ROOT / "exercises" / filename).read_text()
    story.append(Preformatted(source, code_style, maxLineLength=98,
                              splitChars=" ", newLineChars="    "))
    doc = SimpleDocTemplate(str(OUT / f"{slug}.pdf"), pagesize=A4,
                            leftMargin=55, rightMargin=55, topMargin=45, bottomMargin=48,
                            title=topic, author="Mauro Tomas Proto Cassina")
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUT / f"{slug}.pdf")
