"""Test sp80 multiple times to measure win rate with iterative learning."""
import sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(
    __import__("pathlib").Path(__file__).parent.parent / "environment_files"
)
os.environ["ARC_API_KEY"] = "test"
os.environ["RECORDINGS_DIR"] = "recordings"
logging.basicConfig(level=logging.WARNING)

from arc_agi import Arcade, OperationMode
from agents import AVAILABLE_AGENTS

arc = Arcade(
    operation_mode=OperationMode.OFFLINE,
    environments_dir=os.environ["ENVIRONMENTS_DIR"],
)
agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GAME = sys.argv[2] if len(sys.argv) > 2 else "sp80-0ee2d095"
MAX_ACTIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 200

wins = 0
for trial in range(N):
    agent_cls.MAX_ACTIONS = MAX_ACTIONS
    env = arc.make(GAME)
    a = agent_cls(
        card_id=None, game_id=GAME, agent_name="adaptivereasoning",
        ROOT_URL="", record=False, arc_env=env,
    )
    a.main()
    lvl = a.memory.max_level_reached
    states = a.memory.summary()["states_visited"]
    if lvl > 0:
        wins += 1
    print(f"  trial {trial}: level={lvl}, states={states}")
print(f"\nWins: {wins}/{N} on {GAME} ({MAX_ACTIONS} actions)")
