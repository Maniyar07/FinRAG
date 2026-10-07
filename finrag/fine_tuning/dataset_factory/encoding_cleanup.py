from __future__ import annotations

from collections import Counter

from .models import GroundedAnswerTarget, GroundedClaimTarget, MasterExample
from .utils import repair_mojibake


def normalize_supervision(
    examples: list[MasterExample],
) -> tuple[list[MasterExample], dict]:
    output: list[MasterExample] = []
    changed_examples: list[str] = []
    changed_fields: Counter[str] = Counter()
    for example in examples:
        question = repair_mojibake(example.question)
        target = example.target
        answer = repair_mojibake(target.answer)
        claims = [
            GroundedClaimTarget(
                text=repair_mojibake(claim.text), source_ids=claim.source_ids
            )
            for claim in target.claims
        ]
        if question != example.question:
            changed_fields["question"] += 1
        if answer != target.answer:
            changed_fields["answer"] += 1
        changed_fields["claims"] += sum(
            repaired.text != original.text
            for repaired, original in zip(claims, target.claims)
        )
        if question != example.question or answer != target.answer or any(
            repaired.text != original.text
            for repaired, original in zip(claims, target.claims)
        ):
            changed_examples.append(example.example_id)
        normalized_target = GroundedAnswerTarget(
            answer=answer,
            claims=claims,
            source_ids=target.source_ids,
        )
        output.append(
            example.model_copy(
                update={"question": question, "target": normalized_target}
            )
        )
    report = {
        "normalization": "utf8_bytes_misdecoded_as_windows_1252",
        "scope": "supervised_question_and_target_only",
        "examples": len(examples),
        "changed_examples": len(changed_examples),
        "changed_fields": dict(sorted(changed_fields.items())),
        "changed_example_ids": changed_examples,
    }
    return output, report
