import sys
import pygame

from environment import RacingEnvironment, NUM_CARS
from agent import DQNAgent


def train(max_episodes: int = 10_000, render: bool = True) -> None:
    env = RacingEnvironment(render=render)
    agent = DQNAgent(
        state_size=RacingEnvironment.STATE_SIZE,
        action_size=RacingEnvironment.ACTION_SIZE,
    )

    loaded = agent.load()
    start_episode = agent.episode_count + 1 if loaded else 1

    print("=" * 60)
    print("  Autonomous Racing Car - DQN Training")
    print("=" * 60)
    print(f"  Cars:          {env.num_cars}")
    print(f"  State Size:    {RacingEnvironment.STATE_SIZE}")
    print(f"  Action Size:   {RacingEnvironment.ACTION_SIZE}")
    print(f"  Start Episode: {start_episode}")
    print(f"  Device:        {agent.device}")
    print("=" * 60)

    total_steps: int = agent.training_step

    for episode in range(start_episode, max_episodes + 1):
        states = env.reset()
        episode_rewards = [0.0] * env.num_cars
        done_all = False
        step_count = 0

        while not done_all:
            if render:
                if not env.handle_events():
                    agent.save(episode)
                    print("\n[Main] Training stopped by user.")
                    pygame.quit()
                    sys.exit(0)

            actions = agent.select_actions(states)
            next_states, rewards, dones = env.step(actions)

            agent.store_transitions(states, actions, rewards, next_states, dones)
            agent.train_step()
            agent.step_decay_epsilon(env.num_cars)

            for i in range(env.num_cars):
                episode_rewards[i] += rewards[i]

            states = next_states
            total_steps += env.num_cars
            step_count += 1

            if total_steps % 50_000 < env.num_cars:
                agent.episode_count = episode
                agent.high_score = max(agent.high_score, max(episode_rewards))
                agent.save(episode)
                print(
                    f"  [Auto-save] Step {total_steps:,} | "
                    f"Eps: {agent.epsilon:.4f} | "
                    f"Best: {max(episode_rewards):.1f}"
                )

            done_all = step_count >= env.max_steps

            if render:
                avg_reward = sum(episode_rewards) / env.num_cars
                env.draw(
                    episode=episode,
                    high_score=agent.high_score,
                    current_reward=avg_reward,
                    epsilon=agent.epsilon,
                    total_steps=total_steps,
                )

        best_reward = max(episode_rewards)
        avg_reward = sum(episode_rewards) / env.num_cars
        agent.end_episode(episode, best_reward)

        if episode % 10 == 0:
            print(
                f"Episode {episode:5d} | "
                f"Best: {best_reward:8.1f} | "
                f"Avg: {avg_reward:8.1f} | "
                f"High: {agent.high_score:8.1f} | "
                f"Eps: {agent.epsilon:.4f} | "
                f"Buf: {len(agent.memory):6d} | "
                f"Steps: {total_steps}"
            )

    agent.save(max_episodes)
    print("\n[Main] Training complete!")
    if render:
        pygame.quit()


def evaluate(num_episodes: int = 5) -> None:
    env = RacingEnvironment(render=True)
    agent = DQNAgent(
        state_size=RacingEnvironment.STATE_SIZE,
        action_size=RacingEnvironment.ACTION_SIZE,
    )

    if not agent.load():
        print("[Evaluate] No trained model found. Train first!")
        return

    agent.epsilon = 0.0

    print("\n" + "=" * 60)
    print("  Evaluation Mode (No Exploration)")
    print("=" * 60)

    for episode in range(1, num_episodes + 1):
        states = env.reset()
        episode_rewards = [0.0] * env.num_cars
        step_count = 0

        while step_count < env.max_steps:
            if not env.handle_events():
                pygame.quit()
                return

            actions = agent.select_actions(states, training=False)
            next_states, rewards, dones = env.step(actions)

            for i in range(env.num_cars):
                episode_rewards[i] += rewards[i]

            states = next_states
            step_count += 1

            avg_reward = sum(episode_rewards) / env.num_cars
            env.draw(
                episode=episode,
                high_score=agent.high_score,
                current_reward=avg_reward,
                epsilon=0.0,
                total_steps=0,
            )

        best_reward = max(episode_rewards)
        avg_reward = sum(episode_rewards) / env.num_cars
        print(f"Eval Episode {episode}: Best = {best_reward:.1f}, Avg = {avg_reward:.1f}")

    pygame.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN Autonomous Racing Car")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(max_episodes=args.episodes, render=not args.no_render)
    else:
        evaluate()
