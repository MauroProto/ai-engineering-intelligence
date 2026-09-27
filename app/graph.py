from langgraph.graph import StateGraph,START,END
from langchain_core.messages import AIMessage
from .config import settings
from .state import IntelligenceState,RouteDecision,AnalysisArtifact,FinalResponse,Evidence
from .agents.research_agent import make_research_agent
from .agents.analyst_agent import make_analyst_agent
from .hitl import approval_gate
from .observability import tracer,span_input,span_output


def build_graph(gateway,knowledge,checkpointer=None):
    async def supervisor(state):
        with tracer.start_as_current_span("agent.supervisor") as span:
            steps = state.get("steps",0)+1
            span_input(span,{"query":state["query"],"steps":steps},"AGENT")
            if steps > settings.max_agent_steps:
                return {"steps":steps,"next_agent":"finalize","validated":False,
                        "validation_errors":["Presupuesto de pasos agotado"]}
            # El LLM propone la delegacion, la politica impide saltar validacion.
            allowed = ("researcher" if not state.get("research") else
                       "analyst" if not state.get("analysis") else "validate")
            if state.get("validation_errors") and state.get("refinements",0)<1:
                allowed = "researcher"
            decision,usage = await gateway.structured(RouteDecision,
                "Sos supervisor de investigador y analista. Elegi el siguiente especialista "
                "segun los artefactos disponibles. Primero evidencia, despues analisis, "
                "despues validate. Ante errores de validacion, pedir una ronda de refinamiento. "
                "Responde una decision operativa breve, no razonamiento interno.",
                {"query":state["query"],"has_research":bool(state.get("research")),
                 "has_analysis":bool(state.get("analysis")),"errors":state.get("validation_errors",[])})
            # Completar prerequisites aunque el modelo sugiera un salto inseguro.
            chosen = decision.next_agent if decision.next_agent==allowed else allowed
            update = {"steps":steps,"next_agent":chosen,"usages":[usage],
                      "routing_log":[{"step":steps,"proposed":decision.next_agent,
                                      "selected":chosen,"reason":decision.reason}]}
            if state.get("validation_errors") and chosen=="researcher":
                update.update({"refinements":state.get("refinements",0)+1,"analysis":{},
                               "validation_errors":[]})
            span_output(span,update["routing_log"])
            return update

    async def validate(state):
        with tracer.start_as_current_span("validate.final_contract") as span:
            errors = []
            analysis = AnalysisArtifact.model_validate(state["analysis"])
            available = {x["id"] for x in state["evidence"]}
            if set(analysis.source_ids)-available:
                errors.append("Referencias de analista fuera del corpus")
            if analysis.answerable and not analysis.source_ids:
                errors.append("Respuesta afirmativa sin referencias")
            if analysis.answerable and not state["research"]["sufficient"]:
                errors.append("Conflicto: investigador declara evidencia insuficiente")
            if not analysis.answerable and analysis.confidence>.3:
                errors.append("Confianza excesiva para una pregunta no respondible")
            agents = {c["agent"] for c in state["contributions"]}
            if not {"researcher","analyst"}<=agents:
                errors.append("Falta contribucion de un especialista")
            span_input(span,{"analysis":analysis.model_dump()})
            span_output(span,{"passed":not errors,"errors":errors})
            return {"validated":not errors,"validation_errors":errors,
                    "next_agent":"finalize" if not errors or state.get("refinements",0)>=1 else "supervisor"}

    async def finalize(state):
        good = state.get("validated",False)
        if good:
            analysis = AnalysisArtifact.model_validate(state["analysis"])
            result = FinalResponse(answer=analysis.answer,
                references=[Evidence.model_validate(x) for x in state["evidence"] if x["id"] in analysis.source_ids],
                answerable=analysis.answerable,confidence=analysis.confidence,
                limitations=analysis.limitations,contributors=sorted({x["agent"] for x in state["contributions"]}),
                validation_passed=True)
        else:
            result = FinalResponse(answer="No puedo dar una respuesta validada con la evidencia disponible.",
                references=[],answerable=False,confidence=0,limitations="; ".join(state.get("validation_errors",[])),
                contributors=sorted({x["agent"] for x in state.get("contributions",[])}),validation_passed=False)
        return {"final":result.model_dump(),"messages":[AIMessage(content=result.answer)]}

    builder = StateGraph(IntelligenceState)
    for name,node in {"approval":approval_gate,"supervisor":supervisor,
                      "researcher":make_research_agent(gateway,knowledge),
                      "analyst":make_analyst_agent(gateway),"validate":validate,"finalize":finalize}.items():
        builder.add_node(name,node)
    builder.add_edge(START,"approval")
    builder.add_conditional_edges("approval",lambda s:END if s.get("declined") else "supervisor",[END,"supervisor"])
    builder.add_conditional_edges("supervisor",lambda s:s["next_agent"],["researcher","analyst","validate","finalize"])
    builder.add_edge("researcher","supervisor")
    builder.add_edge("analyst","supervisor")
    builder.add_conditional_edges("validate",lambda s:s["next_agent"],["finalize","supervisor"])
    builder.add_edge("finalize",END)
    return builder.compile(checkpointer=checkpointer)
