from langgraph.types import interrupt
from .state import ApprovalRequest


async def approval_gate(state):
    if not state.get("critical_action"):
        return {"approved":True}
    # Nada irreversible antes de interrupt. Se reejecuta al retomar.
    decision = ApprovalRequest.model_validate(interrupt({
        "action":"registrar aprobacion local de accion critica",
        "query":state["query"],"requires_approval":True,
        "warning":"No se ejecuta ninguna accion externa en este laboratorio."}))
    return {"approved":decision.approved,"declined":not decision.approved}
