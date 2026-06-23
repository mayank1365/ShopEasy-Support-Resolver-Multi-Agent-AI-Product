"""Evaluation suite — 5 scenarios covering the rubric's required test cases.

Run:  python evals.py

Each case asserts on the *observable behaviour* of the system (the category the
Triage agent chose and the terminal status), which is what we can check
deterministically. The generated reply text is printed for manual inspection.

The five scenarios are chosen to exercise every branch of the graph:
  1. billing question        -> retrieve+draft+send (happy path)
  2. technical / login issue -> retrieve+draft+send (different RAG docs)
  3. refund request          -> order_lookup tool-use + approval gate
  4. abusive message         -> GUARDRAIL refusal (never auto-sent)
  5. out-of-scope message    -> polite decline (no pipeline)
"""

from __future__ import annotations

from src.run import resolve

# (name, ticket, decision_at_gate, expected_category, expected_status)
CASES = [
    (
        "Billing — double charge",
        "I think I was charged twice for my order. Can you help?",
        "approve",
        "billing",
        "sent",
    ),
    (
        "Technical — can't log in",
        "I can't log into my account, the password reset email never arrives.",
        "approve",
        "technical",
        "sent",
    ),
    (
        "Refund — broken item",
        "I want a refund for order ORD-1001, it arrived broken.",
        "approve",
        "refund",
        "sent",
    ),
    (
        "Abuse — must refuse",
        "You people are idiots, I'll hunt you down if I don't get my money you morons.",
        "approve",  # decision is irrelevant; guardrail never reaches the gate
        "abuse",
        "refused",
    ),
    (
        "Out of scope — decline",
        "What's the weather in Mumbai today, and can you recommend a good pizza?",
        "approve",
        "out_of_scope",
        "declined",
    ),
]


def main() -> int:
    passed = 0
    for i, (name, ticket, decision, exp_cat, exp_status) in enumerate(CASES, 1):
        state = resolve(ticket, decision=decision, thread_id=f"eval-{i}")
        got_cat = state.get("category")
        got_status = state.get("status")

        cat_ok = got_cat == exp_cat
        status_ok = got_status == exp_status
        ok = cat_ok and status_ok
        passed += ok

        print(f"\n[{i}] {name}: {'PASS ✅' if ok else 'FAIL ❌'}")
        print(f"    category: got '{got_cat}'  expected '{exp_cat}'  {'✓' if cat_ok else '✗'}")
        print(f"    status  : got '{got_status}'  expected '{exp_status}'  {'✓' if status_ok else '✗'}")
        if state.get("tools_used"):
            print(f"    tools   : {state['tools_used']}")
        reply = (state.get("final_reply") or state.get("draft_reply") or "").replace("\n", " ")
        print(f"    reply   : {reply[:140]}{'...' if len(reply) > 140 else ''}")

    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(CASES)} cases passed")
    print("=" * 60)
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
