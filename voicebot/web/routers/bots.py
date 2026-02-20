from __future__ import annotations
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import Session

from ..schemas import BotCreateRequest, BotUpdateRequest, IntegrationToolCreateRequest, IntegrationToolUpdateRequest


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/bots")
    def api_list_bots(session: Session = Depends(ctx.get_session)) -> dict:
        bots = ctx.list_bots(session)
        stats = ctx.bots_aggregate_metrics(session)
        items = []
        for b in bots:
            d = ctx._bot_to_dict(b)
            d["stats"] = stats.get(
                b.id,
                {
                    "conversations": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "avg_llm_ttfb_ms": None,
                    "avg_llm_total_ms": None,
                    "avg_total_ms": None,
                },
            )
            items.append(d)
        return {"items": items}

    @router.post("/api/bots")
    def api_create_bot(payload: BotCreateRequest, session: Session = Depends(ctx.get_session)) -> dict:
        default_provider = (ctx._get_app_setting(session, "default_llm_provider") or "").strip().lower()
        default_model = (ctx._get_app_setting(session, "default_llm_model") or "").strip()
        provider = ctx._normalize_llm_provider(payload.llm_provider or default_provider or "openai")
        openai_model = (payload.openai_model or default_model or "o4-mini").strip() or "o4-mini"
        bot = ctx.Bot(
            name=payload.name,
            llm_provider=provider,
            openai_model=openai_model,
            openai_asr_model=(payload.openai_asr_model or "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe",
            web_search_model=(payload.web_search_model or openai_model).strip() or openai_model,
            codex_model=(payload.codex_model or "gpt-5.1-codex-mini").strip() or "gpt-5.1-codex-mini",
            summary_model=(payload.summary_model or openai_model or "gpt-5-nano").strip() or (openai_model or "gpt-5-nano"),
            history_window_turns=int(payload.history_window_turns or 16),
            enable_data_agent=bool(getattr(payload, "enable_data_agent", False)),
            data_agent_api_spec_text=(payload.data_agent_api_spec_text or ""),
            data_agent_auth_json=(payload.data_agent_auth_json or "{}"),
            data_agent_system_prompt=(payload.data_agent_system_prompt or ""),
            data_agent_return_result_directly=bool(getattr(payload, "data_agent_return_result_directly", False)),
            data_agent_prewarm_on_start=bool(getattr(payload, "data_agent_prewarm_on_start", False)),
            data_agent_prewarm_prompt=(payload.data_agent_prewarm_prompt or ""),
            enable_host_actions=bool(getattr(payload, "enable_host_actions", False)),
            enable_host_shell=bool(getattr(payload, "enable_host_shell", False)),
            require_host_action_approval=bool(getattr(payload, "require_host_action_approval", False)),
            system_prompt=payload.system_prompt,
            language=payload.language,
            openai_tts_model=(payload.openai_tts_model or "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts",
            openai_tts_voice=(payload.openai_tts_voice or "alloy").strip() or "alloy",
            openai_tts_speed=float(payload.openai_tts_speed or 1.0),
            start_message_mode=(payload.start_message_mode or "llm").strip() or "llm",
            start_message_text=payload.start_message_text or "",
        )
        ctx.create_bot(session, bot)
        return ctx._bot_to_dict(bot)

    @router.get("/api/bots/{bot_id}")
    def api_get_bot(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx.get_bot(session, bot_id)
        return ctx._bot_to_dict(bot)

    @router.put("/api/bots/{bot_id}")
    def api_update_bot(bot_id: UUID, payload: BotUpdateRequest, session: Session = Depends(ctx.get_session)) -> dict:
        patch = {}
        for k, v in payload.model_dump(exclude_unset=True).items():
            if k == "llm_provider":
                raw = (v or "").strip().lower()
                if raw and raw not in ("openai", "openrouter", "local"):
                    raise ctx.HTTPException(status_code=400, detail="Unsupported LLM provider.")
                patch[k] = raw or "openai"
            elif k in (
                "openai_tts_model",
                "openai_tts_voice",
                "web_search_model",
                "codex_model",
                "summary_model",
                "openai_asr_model",
            ):
                patch[k] = (v or "").strip()
            elif k == "openai_tts_speed":
                patch[k] = float(v) if v is not None else 1.0
            elif k == "history_window_turns":
                try:
                    n = int(v) if v is not None else 16
                except Exception:
                    n = 16
                if n < 1:
                    n = 1
                patch[k] = n
            elif k == "disabled_tools":
                patch[k] = sorted({str(x) for x in (v or []) if str(x).strip()})
            else:
                patch[k] = v
        bot = ctx.update_bot(session, bot_id=bot_id, patch=patch)
        return ctx._bot_to_dict(bot)

    @router.delete("/api/bots/{bot_id}")
    def api_delete_bot(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        ctx.delete_bot(session, bot_id)
        return {"ok": True}

    @router.get("/api/bots/{bot_id}/tools")
    def api_list_tools(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx.get_bot(session, bot_id)
        tools = ctx.list_integration_tools(session, bot_id=bot_id)
        disabled = ctx._disabled_tool_names(bot)
        system_tools = ctx._system_tools_public_list(bot=bot, disabled=disabled)
        return {"items": [ctx._tool_to_dict(t) for t in tools], "system_tools": system_tools}

    @router.post("/api/bots/{bot_id}/tools")
    def api_create_tool(
        bot_id: UUID,
        payload: IntegrationToolCreateRequest,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        bot = ctx.get_bot(session, bot_id)
        tool = ctx.IntegrationTool(
            bot_id=bot.id,
            name=payload.name,
            description=payload.description,
            url=payload.url,
            method=payload.method,
            use_codex_response=payload.use_codex_response,
            enabled=payload.enabled,
            args_required_json=ctx.json.dumps(payload.args_required or [], ensure_ascii=False),
            headers_template_json=payload.headers_template_json,
            request_body_template=payload.request_body_template,
            parameters_schema_json=payload.parameters_schema_json,
            response_schema_json=payload.response_schema_json,
            codex_prompt=payload.codex_prompt,
            postprocess_python=payload.postprocess_python,
            return_result_directly=payload.return_result_directly,
            response_mapper_json=payload.response_mapper_json,
            pagination_json=payload.pagination_json,
            static_reply_template=payload.static_reply_template,
        )
        ctx.create_integration_tool(session, tool)
        return ctx._tool_to_dict(tool)

    @router.put("/api/bots/{bot_id}/tools/{tool_id}")
    def api_update_tool(
        bot_id: UUID,
        tool_id: UUID,
        payload: IntegrationToolUpdateRequest,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        tool = ctx.get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise ctx.HTTPException(status_code=404, detail="Tool not found")
        patch = payload.model_dump(exclude_unset=True)
        if "args_required" in patch:
            patch["args_required_json"] = ctx.json.dumps(patch.pop("args_required") or [], ensure_ascii=False)
        if "headers_template_json" in patch and patch["headers_template_json"] is None:
            patch["headers_template_json"] = "{}"
        tool = ctx.update_integration_tool(session, tool_id, patch)
        return ctx._tool_to_dict(tool)

    @router.delete("/api/bots/{bot_id}/tools/{tool_id}")
    def api_delete_tool(bot_id: UUID, tool_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        tool = ctx.get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise ctx.HTTPException(status_code=404, detail="Tool not found")
        ctx.delete_integration_tool(session, tool_id)
        return {"ok": True}

    app.include_router(router)
