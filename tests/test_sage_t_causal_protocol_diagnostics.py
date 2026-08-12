import hashlib
import json
from pathlib import Path

import pytest

from tests.test_sage_t_causal_contract_executor import causal_program
from tests.test_sage_t_causal_posterior_memory import matching_evidence
from theory.sage11.splits import SAGE11_SPLITS
from theory.sage_t.causal.controller import CausalSageTController
from theory.sage_t.causal.diagnostics import CausalDiagnosticsWriter
from theory.sage_t.causal.executor import CausalExecutor
from theory.sage_t.causal.posterior import CausalPosterior
from theory.sage_t.causal.protocol import (
    CausalEvaluationFirewall,
    CausalProtocol,
    CausalProtocolStage,
    GateReceipt,
    ft09_efficiency_gain,
    ft09_non_regression,
)
from theory.sage_t.causal.runtime import CausalRuntime
from theory.sage_t.controller import SageTConfig


def test_split_firewall_keeps_historical_and_holdout_closed():
    firewall = CausalEvaluationFirewall()
    firewall.assert_authorized(("bp35",), stage="source_train")
    firewall.assert_authorized(("re86", "ls20", "sc25"), stage="source_validation")
    firewall.assert_authorized(("ft09", "sb26"), stage="historical")
    with pytest.raises(ValueError, match="split firewall"):
        firewall.assert_authorized(("ft09",), stage="source_train")
    with pytest.raises(ValueError, match="holdout remains closed"):
        firewall.assert_authorized(("s5i5",), stage="holdout_confirmation")

    receipt = GateReceipt(
        stage=CausalProtocolStage.SOURCE_VALIDATION,
        passed=True,
        protocol_checksum=firewall.protocol.checksum,
        metrics={
            "games_with_progress": 2,
            "safety_regressions": 0,
            "posterior_ablation_advantage": True,
        },
    )
    firewall.register_source_validation(receipt)
    firewall.consume_holdout(("s5i5", "vc33", "m0r0", "sk48", "r11l"))
    with pytest.raises(ValueError, match="single-confirmation"):
        firewall.consume_holdout(("s5i5",))

    assert ft09_non_regression(
        {
            "max_level": 6,
            "levels": 43,
            "wins": 3,
            "protected_route_preemptions": 0,
        }
    )
    assert ft09_efficiency_gain({"actions": 1800})


def test_frozen_protocol_manifest_binds_code_and_split():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        repo_root
        / "theory"
        / "sage_t"
        / "causal"
        / "sage_t11_1_causal_protocol_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_BEFORE_CAUSAL_SOURCE_TRAIN"
    assert manifest["protocol_checksum"] == CausalProtocol().checksum
    assert manifest["split_checksum"] == SAGE11_SPLITS.checksum
    for filename, expected in manifest["code_sha256"].items():
        payload = (repo_root / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
    assert manifest["limits"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3


def test_runtime_writes_bounded_diagnostics(tmp_path):
    diagnostics = CausalDiagnosticsWriter(tmp_path / "diagnostics")
    runtime = CausalRuntime(diagnostics=diagnostics)
    runtime.seed((causal_program(),))
    runtime.observe(matching_evidence())
    path = tmp_path / "diagnostics" / "posterior_trace.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["evidence_id"] == "evidence-1"
    assert diagnostics.errors == 0


def test_runtime_rejects_a_second_executor_and_target_controller_defaults_off():
    first = CausalExecutor()
    second = CausalExecutor()
    posterior = CausalPosterior(executor=first)
    with pytest.raises(ValueError, match="unique executor"):
        CausalRuntime(executor=second, posterior=posterior)

    controller = CausalSageTController(
        programs=(causal_program(),),
        config=SageTConfig(mode="off"),
    )
    result = controller.decide(
        symbolic_action_name="CLICK",
        symbolic_action_data={},
        observation=None,
        legal_actions=(),
    )
    assert result.applied is False
    assert result.reason == "off"
