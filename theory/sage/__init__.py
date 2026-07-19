"""SAGE closed-loop integration scaffolds."""

__all__ = [
    "DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH",
    "DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH",
    "DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH",
    "DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH",
    "DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH",
    "DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH",
    "DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH",
    "DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH",
    "DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH",
    "DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH",
    "DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH",
    "DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH",
    "DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH",
    "DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH",
    "DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH",
    "DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH",
    "DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH",
    "DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH",
    "DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH",
    "DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH",
    "DEFAULT_SAGE6B_M3_EXECUTION_PATH",
    "DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH",
    "DEFAULT_SAGE6D_HANDOFF_PATH",
    "DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH",
    "DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH",
    "DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH",
    "DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH",
    "DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH",
    "DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH",
    "DEFAULT_SAGE7D_A32_HANDOFF_PATH",
    "DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH",
    "DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH",
    "DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH",
    "A32ReviewCandidateItem",
    "PolicyActionOption",
    "RelationalMemoryPolicyEntry",
    "apply_relational_memory_policy",
    "build_relational_memory_policy_entries",
    "collect_live_prefix_counterfactual",
    "decide_subgoal_switch",
    "evaluate_progress_stall_trigger",
    "run_sage0_known_game_scaffold",
    "run_sage1_known_game_closed_loop",
    "run_sage2_known_game_policy_probe",
    "run_sage3_subgoal_switch_probe",
    "run_sage4_long_horizon_transfer",
    "run_sage4b_progress_stall_trigger_probe",
    "run_sage4c_long_horizon_progress_stall_transfer",
    "run_sage5_unknown_game_bounded_probe",
    "run_sage5b_switch_attribution_placeholder_audit",
    "run_sage5c_live_mini_frontier_generation",
    "run_sage5d_live_mini_frontier_m3_execution",
    "run_sage5e_distributed_live_mini_frontier_generation",
    "run_sage5f_mini_frontier_event_consolidation",
    "run_sage5g_a32_review_handoff",
    "run_sage5h_controlled_followup_acquisition",
    "run_sage5i_control_surface_expansion",
    "run_sage5j_parameterized_control_acquisition",
    "run_sage6_second_unknown_game_transfer",
    "run_sage6a_switch_attribution_mini_frontier",
    "run_sage6b_second_unknown_game_m3_execution",
    "run_sage6c_second_unknown_game_event_consolidation",
    "run_sage6d_second_unknown_game_handoff",
    "run_sage6e_second_unknown_game_followup_execution",
    "run_sage6f_second_unknown_game_control_dependence_consolidation",
    "run_sage7_third_unknown_game_transfer",
    "run_sage7a_parameterized_mini_frontier",
    "run_sage7b_parameterized_execution",
    "run_sage7c_parameterized_consolidation",
    "run_sage7d_a32_handoff",
    "run_sage8a_relational_memory_policy_integration",
    "run_sage8b_relational_memory_ab_evaluation",
    "run_sage8c_relational_memory_multi_action_evaluation",
    "write_sage0_known_game_scaffold",
    "write_sage1_known_game_results",
    "write_sage2_policy_probe_results",
    "write_sage3_subgoal_switch_results",
    "write_sage4_long_horizon_results",
    "write_sage4b_progress_stall_results",
    "write_sage4c_long_horizon_progress_stall_results",
    "write_sage5_unknown_game_bounded_probe_results",
    "write_sage5b_switch_attribution_placeholder_audit",
    "write_sage5c_live_mini_frontier_generation",
    "write_sage5d_live_mini_frontier_m3_results",
    "write_sage5e_distributed_live_mini_frontier_results",
    "write_sage5f_mini_frontier_event_consolidation",
    "write_sage5g_a32_review_handoff",
    "write_sage5h_controlled_followup_acquisition",
    "write_sage5i_control_surface_expansion",
    "write_sage5j_parameterized_control_acquisition",
    "write_sage6_second_unknown_game_transfer",
    "write_sage6a_switch_attribution_mini_frontier",
    "write_sage6b_second_unknown_game_m3_execution",
    "write_sage6c_second_unknown_game_event_consolidation",
    "write_sage6d_second_unknown_game_handoff",
    "write_sage6e_second_unknown_game_followup_execution",
    "write_sage6f_second_unknown_game_control_dependence_consolidation",
    "write_sage7_third_unknown_game_transfer",
    "write_sage7a_parameterized_mini_frontier",
    "write_sage7b_parameterized_execution",
    "write_sage7c_parameterized_consolidation",
    "write_sage7d_a32_handoff",
    "write_sage8a_relational_memory_policy",
    "write_sage8b_relational_memory_ab_evaluation",
    "write_sage8c_relational_memory_multi_action_evaluation",
]


def __getattr__(name: str):
    if name in {
        "DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH",
        "run_sage0_known_game_scaffold",
        "write_sage0_known_game_scaffold",
    }:
        from .known_game_closed_loop_scaffold import (
            DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH,
            run_sage0_known_game_scaffold,
            write_sage0_known_game_scaffold,
        )

        return {
            "DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH": (
                DEFAULT_SAGE0_KNOWN_GAME_SCAFFOLD_OUTPUT_PATH
            ),
            "run_sage0_known_game_scaffold": run_sage0_known_game_scaffold,
            "write_sage0_known_game_scaffold": write_sage0_known_game_scaffold,
        }[name]
    if name in {
        "DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH",
        "run_sage1_known_game_closed_loop",
        "write_sage1_known_game_results",
    }:
        from .known_game_closed_loop_runner import (
            DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH,
            run_sage1_known_game_closed_loop,
            write_sage1_known_game_results,
        )

        return {
            "DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH": (
                DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH
            ),
            "run_sage1_known_game_closed_loop": run_sage1_known_game_closed_loop,
            "write_sage1_known_game_results": write_sage1_known_game_results,
        }[name]
    if name == "collect_live_prefix_counterfactual":
        from .live_prefix_counterfactual_collector import (
            collect_live_prefix_counterfactual,
        )

        return collect_live_prefix_counterfactual
    if name == "DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH":
        from .policy_loop_guard import DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH

        return DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH
    if name in {
        "DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH",
        "run_sage2_known_game_policy_probe",
        "write_sage2_policy_probe_results",
    }:
        from .known_game_policy_probe import (
            DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH,
            run_sage2_known_game_policy_probe,
            write_sage2_policy_probe_results,
        )

        return {
            "DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH": (
                DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH
            ),
            "run_sage2_known_game_policy_probe": run_sage2_known_game_policy_probe,
            "write_sage2_policy_probe_results": write_sage2_policy_probe_results,
        }[name]
    if name in {
        "DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH",
        "decide_subgoal_switch",
        "run_sage3_subgoal_switch_probe",
        "write_sage3_subgoal_switch_results",
    }:
        from .subgoal_switcher import (
            DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH,
            decide_subgoal_switch,
            run_sage3_subgoal_switch_probe,
            write_sage3_subgoal_switch_results,
        )

        return {
            "DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH": (
                DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH
            ),
            "decide_subgoal_switch": decide_subgoal_switch,
            "run_sage3_subgoal_switch_probe": run_sage3_subgoal_switch_probe,
            "write_sage3_subgoal_switch_results": write_sage3_subgoal_switch_results,
        }[name]
    if name in {
        "DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH",
        "run_sage4_long_horizon_transfer",
        "write_sage4_long_horizon_results",
    }:
        from .long_horizon_transfer import (
            DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH,
            run_sage4_long_horizon_transfer,
            write_sage4_long_horizon_results,
        )

        return {
            "DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH": (
                DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH
            ),
            "run_sage4_long_horizon_transfer": run_sage4_long_horizon_transfer,
            "write_sage4_long_horizon_results": write_sage4_long_horizon_results,
        }[name]
    if name in {
        "DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH",
        "evaluate_progress_stall_trigger",
        "run_sage4b_progress_stall_trigger_probe",
        "write_sage4b_progress_stall_results",
    }:
        from .progress_stall_trigger import (
            DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH,
            evaluate_progress_stall_trigger,
            run_sage4b_progress_stall_trigger_probe,
            write_sage4b_progress_stall_results,
        )

        return {
            "DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH": (
                DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH
            ),
            "evaluate_progress_stall_trigger": evaluate_progress_stall_trigger,
            "run_sage4b_progress_stall_trigger_probe": (
                run_sage4b_progress_stall_trigger_probe
            ),
            "write_sage4b_progress_stall_results": (
                write_sage4b_progress_stall_results
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH",
        "run_sage4c_long_horizon_progress_stall_transfer",
        "write_sage4c_long_horizon_progress_stall_results",
    }:
        from .long_horizon_progress_stall_transfer import (
            DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH,
            run_sage4c_long_horizon_progress_stall_transfer,
            write_sage4c_long_horizon_progress_stall_results,
        )

        return {
            "DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH": (
                DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH
            ),
            "run_sage4c_long_horizon_progress_stall_transfer": (
                run_sage4c_long_horizon_progress_stall_transfer
            ),
            "write_sage4c_long_horizon_progress_stall_results": (
                write_sage4c_long_horizon_progress_stall_results
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH",
        "run_sage5_unknown_game_bounded_probe",
        "write_sage5_unknown_game_bounded_probe_results",
    }:
        from .unknown_game_bounded_probe import (
            DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
            run_sage5_unknown_game_bounded_probe,
            write_sage5_unknown_game_bounded_probe_results,
        )

        return {
            "DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH": (
                DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH
            ),
            "run_sage5_unknown_game_bounded_probe": (
                run_sage5_unknown_game_bounded_probe
            ),
            "write_sage5_unknown_game_bounded_probe_results": (
                write_sage5_unknown_game_bounded_probe_results
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH",
        "run_sage5b_switch_attribution_placeholder_audit",
        "write_sage5b_switch_attribution_placeholder_audit",
    }:
        from .switch_attribution_placeholder_audit import (
            DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH,
            run_sage5b_switch_attribution_placeholder_audit,
            write_sage5b_switch_attribution_placeholder_audit,
        )

        return {
            "DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH": (
                DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH
            ),
            "run_sage5b_switch_attribution_placeholder_audit": (
                run_sage5b_switch_attribution_placeholder_audit
            ),
            "write_sage5b_switch_attribution_placeholder_audit": (
                write_sage5b_switch_attribution_placeholder_audit
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH",
        "run_sage5c_live_mini_frontier_generation",
        "write_sage5c_live_mini_frontier_generation",
    }:
        from .live_mini_frontier_generation import (
            DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH,
            run_sage5c_live_mini_frontier_generation,
            write_sage5c_live_mini_frontier_generation,
        )

        return {
            "DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH": (
                DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH
            ),
            "run_sage5c_live_mini_frontier_generation": (
                run_sage5c_live_mini_frontier_generation
            ),
            "write_sage5c_live_mini_frontier_generation": (
                write_sage5c_live_mini_frontier_generation
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH",
        "run_sage5d_live_mini_frontier_m3_execution",
        "write_sage5d_live_mini_frontier_m3_results",
    }:
        from .live_mini_frontier_m3_executor import (
            DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH,
            run_sage5d_live_mini_frontier_m3_execution,
            write_sage5d_live_mini_frontier_m3_results,
        )

        return {
            "DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH": (
                DEFAULT_SAGE5D_LIVE_MINI_FRONTIER_M3_RESULTS_PATH
            ),
            "run_sage5d_live_mini_frontier_m3_execution": (
                run_sage5d_live_mini_frontier_m3_execution
            ),
            "write_sage5d_live_mini_frontier_m3_results": (
                write_sage5d_live_mini_frontier_m3_results
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH",
        "run_sage5e_distributed_live_mini_frontier_generation",
        "write_sage5e_distributed_live_mini_frontier_results",
    }:
        from .distributed_live_mini_frontier_generation import (
            DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH,
            run_sage5e_distributed_live_mini_frontier_generation,
            write_sage5e_distributed_live_mini_frontier_results,
        )

        return {
            "DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH": (
                DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH
            ),
            "run_sage5e_distributed_live_mini_frontier_generation": (
                run_sage5e_distributed_live_mini_frontier_generation
            ),
            "write_sage5e_distributed_live_mini_frontier_results": (
                write_sage5e_distributed_live_mini_frontier_results
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH",
        "run_sage5f_mini_frontier_event_consolidation",
        "write_sage5f_mini_frontier_event_consolidation",
    }:
        from .mini_frontier_event_consolidation import (
            DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH,
            run_sage5f_mini_frontier_event_consolidation,
            write_sage5f_mini_frontier_event_consolidation,
        )

        return {
            "DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH": (
                DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH
            ),
            "run_sage5f_mini_frontier_event_consolidation": (
                run_sage5f_mini_frontier_event_consolidation
            ),
            "write_sage5f_mini_frontier_event_consolidation": (
                write_sage5f_mini_frontier_event_consolidation
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH",
        "A32ReviewCandidateItem",
        "run_sage5g_a32_review_handoff",
        "write_sage5g_a32_review_handoff",
    }:
        from .a32_review_handoff import (
            DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH,
            A32ReviewCandidateItem,
            run_sage5g_a32_review_handoff,
            write_sage5g_a32_review_handoff,
        )

        return {
            "DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH": (
                DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH
            ),
            "A32ReviewCandidateItem": A32ReviewCandidateItem,
            "run_sage5g_a32_review_handoff": run_sage5g_a32_review_handoff,
            "write_sage5g_a32_review_handoff": write_sage5g_a32_review_handoff,
        }[name]
    if name in {
        "DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH",
        "run_sage5h_controlled_followup_acquisition",
        "write_sage5h_controlled_followup_acquisition",
    }:
        from .controlled_followup_acquisition import (
            DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH,
            run_sage5h_controlled_followup_acquisition,
            write_sage5h_controlled_followup_acquisition,
        )

        return {
            "DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH": (
                DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH
            ),
            "run_sage5h_controlled_followup_acquisition": (
                run_sage5h_controlled_followup_acquisition
            ),
            "write_sage5h_controlled_followup_acquisition": (
                write_sage5h_controlled_followup_acquisition
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH",
        "run_sage5i_control_surface_expansion",
        "write_sage5i_control_surface_expansion",
    }:
        from .control_surface_expansion import (
            DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH,
            run_sage5i_control_surface_expansion,
            write_sage5i_control_surface_expansion,
        )

        return {
            "DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH": (
                DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH
            ),
            "run_sage5i_control_surface_expansion": (
                run_sage5i_control_surface_expansion
            ),
            "write_sage5i_control_surface_expansion": (
                write_sage5i_control_surface_expansion
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH",
        "run_sage5j_parameterized_control_acquisition",
        "write_sage5j_parameterized_control_acquisition",
    }:
        from .parameterized_control_acquisition import (
            DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH,
            run_sage5j_parameterized_control_acquisition,
            write_sage5j_parameterized_control_acquisition,
        )

        return {
            "DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH": (
                DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH
            ),
            "run_sage5j_parameterized_control_acquisition": (
                run_sage5j_parameterized_control_acquisition
            ),
            "write_sage5j_parameterized_control_acquisition": (
                write_sage5j_parameterized_control_acquisition
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH",
        "run_sage6_second_unknown_game_transfer",
        "write_sage6_second_unknown_game_transfer",
    }:
        from .second_unknown_game_transfer import (
            DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH,
            run_sage6_second_unknown_game_transfer,
            write_sage6_second_unknown_game_transfer,
        )

        return {
            "DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH": (
                DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH
            ),
            "run_sage6_second_unknown_game_transfer": (
                run_sage6_second_unknown_game_transfer
            ),
            "write_sage6_second_unknown_game_transfer": (
                write_sage6_second_unknown_game_transfer
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH",
        "run_sage6a_switch_attribution_mini_frontier",
        "write_sage6a_switch_attribution_mini_frontier",
    }:
        from .second_unknown_game_switch_frontier import (
            DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
            run_sage6a_switch_attribution_mini_frontier,
            write_sage6a_switch_attribution_mini_frontier,
        )

        return {
            "DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH": (
                DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH
            ),
            "run_sage6a_switch_attribution_mini_frontier": (
                run_sage6a_switch_attribution_mini_frontier
            ),
            "write_sage6a_switch_attribution_mini_frontier": (
                write_sage6a_switch_attribution_mini_frontier
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6B_M3_EXECUTION_PATH",
        "run_sage6b_second_unknown_game_m3_execution",
        "write_sage6b_second_unknown_game_m3_execution",
    }:
        from .second_unknown_game_m3_execution import (
            DEFAULT_SAGE6B_M3_EXECUTION_PATH,
            run_sage6b_second_unknown_game_m3_execution,
            write_sage6b_second_unknown_game_m3_execution,
        )

        return {
            "DEFAULT_SAGE6B_M3_EXECUTION_PATH": DEFAULT_SAGE6B_M3_EXECUTION_PATH,
            "run_sage6b_second_unknown_game_m3_execution": (
                run_sage6b_second_unknown_game_m3_execution
            ),
            "write_sage6b_second_unknown_game_m3_execution": (
                write_sage6b_second_unknown_game_m3_execution
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH",
        "run_sage6c_second_unknown_game_event_consolidation",
        "write_sage6c_second_unknown_game_event_consolidation",
    }:
        from .second_unknown_game_event_consolidation import (
            DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH,
            run_sage6c_second_unknown_game_event_consolidation,
            write_sage6c_second_unknown_game_event_consolidation,
        )

        return {
            "DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH": (
                DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH
            ),
            "run_sage6c_second_unknown_game_event_consolidation": (
                run_sage6c_second_unknown_game_event_consolidation
            ),
            "write_sage6c_second_unknown_game_event_consolidation": (
                write_sage6c_second_unknown_game_event_consolidation
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6D_HANDOFF_PATH",
        "run_sage6d_second_unknown_game_handoff",
        "write_sage6d_second_unknown_game_handoff",
    }:
        from .second_unknown_game_handoff_compiler import (
            DEFAULT_SAGE6D_HANDOFF_PATH,
            run_sage6d_second_unknown_game_handoff,
            write_sage6d_second_unknown_game_handoff,
        )

        return {
            "DEFAULT_SAGE6D_HANDOFF_PATH": DEFAULT_SAGE6D_HANDOFF_PATH,
            "run_sage6d_second_unknown_game_handoff": (
                run_sage6d_second_unknown_game_handoff
            ),
            "write_sage6d_second_unknown_game_handoff": (
                write_sage6d_second_unknown_game_handoff
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH",
        "run_sage6e_second_unknown_game_followup_execution",
        "write_sage6e_second_unknown_game_followup_execution",
    }:
        from .second_unknown_game_followup_execution import (
            DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH,
            run_sage6e_second_unknown_game_followup_execution,
            write_sage6e_second_unknown_game_followup_execution,
        )

        return {
            "DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH": (
                DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH
            ),
            "run_sage6e_second_unknown_game_followup_execution": (
                run_sage6e_second_unknown_game_followup_execution
            ),
            "write_sage6e_second_unknown_game_followup_execution": (
                write_sage6e_second_unknown_game_followup_execution
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH",
        "run_sage6f_second_unknown_game_control_dependence_consolidation",
        "write_sage6f_second_unknown_game_control_dependence_consolidation",
    }:
        from .second_unknown_game_control_dependence_consolidation import (
            DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH,
            run_sage6f_second_unknown_game_control_dependence_consolidation,
            write_sage6f_second_unknown_game_control_dependence_consolidation,
        )

        return {
            "DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH": (
                DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH
            ),
            "run_sage6f_second_unknown_game_control_dependence_consolidation": (
                run_sage6f_second_unknown_game_control_dependence_consolidation
            ),
            "write_sage6f_second_unknown_game_control_dependence_consolidation": (
                write_sage6f_second_unknown_game_control_dependence_consolidation
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH",
        "run_sage7_third_unknown_game_transfer",
        "write_sage7_third_unknown_game_transfer",
    }:
        from .third_unknown_game_transfer import (
            DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH,
            run_sage7_third_unknown_game_transfer,
            write_sage7_third_unknown_game_transfer,
        )

        return {
            "DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH": (
                DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH
            ),
            "run_sage7_third_unknown_game_transfer": (
                run_sage7_third_unknown_game_transfer
            ),
            "write_sage7_third_unknown_game_transfer": (
                write_sage7_third_unknown_game_transfer
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH",
        "run_sage7a_parameterized_mini_frontier",
        "write_sage7a_parameterized_mini_frontier",
    }:
        from .third_unknown_game_parameterized_frontier import (
            DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
            run_sage7a_parameterized_mini_frontier,
            write_sage7a_parameterized_mini_frontier,
        )

        return {
            "DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH": (
                DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH
            ),
            "run_sage7a_parameterized_mini_frontier": (
                run_sage7a_parameterized_mini_frontier
            ),
            "write_sage7a_parameterized_mini_frontier": (
                write_sage7a_parameterized_mini_frontier
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH",
        "run_sage7b_parameterized_execution",
        "write_sage7b_parameterized_execution",
    }:
        from .third_unknown_game_parameterized_execution import (
            DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH,
            run_sage7b_parameterized_execution,
            write_sage7b_parameterized_execution,
        )

        return {
            "DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH": (
                DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH
            ),
            "run_sage7b_parameterized_execution": (run_sage7b_parameterized_execution),
            "write_sage7b_parameterized_execution": (
                write_sage7b_parameterized_execution
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH",
        "run_sage7c_parameterized_consolidation",
        "write_sage7c_parameterized_consolidation",
    }:
        from .third_unknown_game_parameterized_consolidation import (
            DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH,
            run_sage7c_parameterized_consolidation,
            write_sage7c_parameterized_consolidation,
        )

        return {
            "DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH": (
                DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH
            ),
            "run_sage7c_parameterized_consolidation": (
                run_sage7c_parameterized_consolidation
            ),
            "write_sage7c_parameterized_consolidation": (
                write_sage7c_parameterized_consolidation
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE7D_A32_HANDOFF_PATH",
        "run_sage7d_a32_handoff",
        "write_sage7d_a32_handoff",
    }:
        from .third_unknown_game_a32_handoff import (
            DEFAULT_SAGE7D_A32_HANDOFF_PATH,
            run_sage7d_a32_handoff,
            write_sage7d_a32_handoff,
        )

        return {
            "DEFAULT_SAGE7D_A32_HANDOFF_PATH": DEFAULT_SAGE7D_A32_HANDOFF_PATH,
            "run_sage7d_a32_handoff": run_sage7d_a32_handoff,
            "write_sage7d_a32_handoff": write_sage7d_a32_handoff,
        }[name]
    if name in {
        "DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH",
        "PolicyActionOption",
        "RelationalMemoryPolicyEntry",
        "apply_relational_memory_policy",
        "build_relational_memory_policy_entries",
        "run_sage8a_relational_memory_policy_integration",
        "write_sage8a_relational_memory_policy",
    }:
        from .relational_memory_policy import (
            DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
            PolicyActionOption,
            RelationalMemoryPolicyEntry,
            apply_relational_memory_policy,
            build_relational_memory_policy_entries,
            run_sage8a_relational_memory_policy_integration,
            write_sage8a_relational_memory_policy,
        )

        return {
            "DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH": (
                DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH
            ),
            "PolicyActionOption": PolicyActionOption,
            "RelationalMemoryPolicyEntry": RelationalMemoryPolicyEntry,
            "apply_relational_memory_policy": apply_relational_memory_policy,
            "build_relational_memory_policy_entries": (
                build_relational_memory_policy_entries
            ),
            "run_sage8a_relational_memory_policy_integration": (
                run_sage8a_relational_memory_policy_integration
            ),
            "write_sage8a_relational_memory_policy": (
                write_sage8a_relational_memory_policy
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH",
        "run_sage8b_relational_memory_ab_evaluation",
        "write_sage8b_relational_memory_ab_evaluation",
    }:
        from .relational_memory_ab_evaluation import (
            DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
            run_sage8b_relational_memory_ab_evaluation,
            write_sage8b_relational_memory_ab_evaluation,
        )

        return {
            "DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH": (
                DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH
            ),
            "run_sage8b_relational_memory_ab_evaluation": (
                run_sage8b_relational_memory_ab_evaluation
            ),
            "write_sage8b_relational_memory_ab_evaluation": (
                write_sage8b_relational_memory_ab_evaluation
            ),
        }[name]
    if name in {
        "DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH",
        "run_sage8c_relational_memory_multi_action_evaluation",
        "write_sage8c_relational_memory_multi_action_evaluation",
    }:
        from .relational_memory_multi_action_evaluation import (
            DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH,
            run_sage8c_relational_memory_multi_action_evaluation,
            write_sage8c_relational_memory_multi_action_evaluation,
        )

        return {
            "DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH": (
                DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH
            ),
            "run_sage8c_relational_memory_multi_action_evaluation": (
                run_sage8c_relational_memory_multi_action_evaluation
            ),
            "write_sage8c_relational_memory_multi_action_evaluation": (
                write_sage8c_relational_memory_multi_action_evaluation
            ),
        }[name]
    raise AttributeError(name)
