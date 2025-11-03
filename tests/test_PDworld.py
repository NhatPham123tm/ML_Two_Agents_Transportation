import os, random, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import world

def run_pdworld_demo():
    env = world.PDWorld(
        H=5, W=5,
        obstacles={(2,2)},
        pickups={(0,0): 2, (4,0): 1},
        drops=[(4,4)],
        start_F=(0,1),
        start_M=(4,1),
    )

    print("Initial state F:", env.pos_F, "M:", env.pos_M)
    rng = random.Random(42)

    # Run 20 alternating actions (F then M)
    for t in range(20):
        agent = "F" if t % 2 == 0 else "M"
        acts = env.applicable_actions(agent)
        if not acts:
            print(f"{agent}: no valid actions, skipping")
            continue
        act = rng.choice(acts)
        _, r, info = env.step(agent, act)
        print(f"Step {t:02d} | Agent {agent:1s} | Act {act:8s} | "
              f"Reward {r:+.2f} | PosF={info['pos_F']} PosM={info['pos_M']} "
              f"CarryF={info['carry_F']} CarryM={info['carry_M']}")
        if info["terminal"]:
            print("Reached terminal state at step", t)
            break

    print("Episode finished. Remaining blocks:", env._blocks)

if __name__ == "__main__":
    run_pdworld_demo()
