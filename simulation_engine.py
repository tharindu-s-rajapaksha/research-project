import sys
import argparse
import collections
import time
import os

import numpy as np
import torch
import gymnasium as gym

# Mute pygame prompt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import config as cfg
from meta_agent import HormonalMetaAgent
from worker import LocalRLWorker
from environments import VolatileBandit, HighStakesForaging
import experiments
import evaluation
from ablation import run_ablation_study, plot_comparative_bars, print_scientific_conclusions

# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------
C_BG = (30, 30, 35)
C_PANEL = (45, 45, 55)
C_TEXT = (220, 220, 220)
C_DA = (255, 215, 0)
C_NA = (220, 20, 60)
C_5HT = (60, 179, 113)
C_ALPHA = (148, 0, 211)
C_TAU = (255, 140, 0)
C_GAMMA = (0, 128, 128)

# ---------------------------------------------------------
# Pygame Scrolling Plot
# ---------------------------------------------------------
class ScrollingPlot:
    def __init__(self, x, y, w, h, title, labels, colors, y_range=(0, 1), max_pts=200):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.labels = labels
        self.colors = colors
        self.y_range = list(y_range)
        self.max_pts = max_pts
        self.data = [collections.deque(maxlen=max_pts) for _ in labels]
        self.font = pygame.font.SysFont("Consolas", 12)
        self.title_font = pygame.font.SysFont("Consolas", 14, bold=True)
        
    def add_data(self, vals):
        for i, v in enumerate(vals):
            self.data[i].append(v)
            
    def draw(self, surface):
        pygame.draw.rect(surface, C_PANEL, self.rect)
        pygame.draw.rect(surface, (100, 100, 100), self.rect, 1)
        
        # Title
        t_surf = self.title_font.render(self.title, True, C_TEXT)
        surface.blit(t_surf, (self.rect.x + 5, self.rect.y + 5))
        
        # Legend & Current Values
        lx = self.rect.x + 5
        ly = self.rect.y + 25
        for i, label in enumerate(self.labels):
            pygame.draw.rect(surface, self.colors[i], (lx, ly + 2, 8, 8))
            
            # Get latest value
            val_str = ""
            if len(self.data[i]) > 0:
                val_str = f": {self.data[i][-1]:.3f}"
            
            l_surf = self.font.render(f"{label}{val_str}", True, C_TEXT)
            surface.blit(l_surf, (lx + 15, ly))
            lx += l_surf.get_width() + 25
            
        # Determine global Y bounds first
        all_vals = [v for dq in self.data for v in dq]
        min_y = min(self.y_range[0], min(all_vals) if all_vals else self.y_range[0])
        max_y = max(self.y_range[1], max(all_vals) if all_vals else self.y_range[1])
        if min_y == max_y: max_y = min_y + 1

        # Draw Grids with Values
        grid_color = (70, 70, 80)
        n_grids = 4
        for k in range(n_grids):
            val = min_y + (max_y - min_y) * (k / (n_grids - 1))
            norm = (val - min_y) / (max_y - min_y + 1e-6)
            py = self.rect.y + self.rect.height - 5 - norm * (self.rect.height - 40)
            
            # Draw line
            pygame.draw.line(surface, grid_color, (self.rect.x, py), (self.rect.x + self.rect.width, py), 1)
            
            # Draw text label on the right side
            val_surf = self.font.render(f"{val:.1f}", True, (150, 150, 150))
            surface.blit(val_surf, (self.rect.x + self.rect.width - val_surf.get_width() - 5, py - 14))

        # Lines
        for i, q in enumerate(self.data):
            if len(q) < 2: continue
            pts = []
            for j, val in enumerate(q):
                px = self.rect.x + (j / (self.max_pts - 1)) * self.rect.width
                val = max(min_y, min(max_y, val))
                norm = (val - min_y) / (max_y - min_y + 1e-6)
                py = self.rect.y + self.rect.height - 5 - norm * (self.rect.height - 40)
                pts.append((px, py))
            pygame.draw.lines(surface, self.colors[i], False, pts, 2)

# ---------------------------------------------------------
# Live Engine
# ---------------------------------------------------------
class SimulationEngine:
    def __init__(self, exp_id: int):
        self.exp_id = exp_id
        self.speed = 5  # Steps per second
        self.paused = False
        self.step_accumulator = 0.0
        
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Experiment {exp_id} - Live Simulation")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("Consolas", 14)
        self.large_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.huge_font = pygame.font.SysFont("Consolas", 36, bold=True)
        
        # Dashboard Layout
        # Story View: Top Half
        self.story_rect = pygame.Rect(10, 40, 1180, 400)
        # Metrics: Bottom Half
        self.plot_hormones = ScrollingPlot(10, 450, 580, 160, "Hormone Concentrations", 
                                           ["DA_eff", "NA", "5HT"], [C_DA, C_NA, C_5HT], y_range=(0, 5))
        self.plot_hyperparams = ScrollingPlot(610, 450, 580, 160, "Dynamic Hyperparameters", 
                                              ["Alpha (LR)", "Tau (Temp)", "Gamma (Disc)"], [C_ALPHA, C_TAU, C_GAMMA], y_range=(0, 2))
        self.plot_rewards = ScrollingPlot(10, 620, 1180, 160, "Instantaneous Reward", 
                                          ["Reward"], [(50, 150, 255)], y_range=(-10, 50))
        
        self.setup_experiment()
        
    def setup_experiment(self):
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        
        self.ablation_cfg = cfg.ABLATION_CONFIGS["Full Model"]
        self.meta = HormonalMetaAgent(
            enable_da=self.ablation_cfg["DA"],
            enable_na=self.ablation_cfg["NA"],
            enable_5ht=self.ablation_cfg["5HT"],
        )
        
        if self.exp_id == 1:
            self.env = VolatileBandit(seed=cfg.SEED)
            self.worker = LocalRLWorker(self.env.observation_dim, self.env.action_dim)
            self.state = self.env.reset()
            self.step_i = 0
            self.total_steps = cfg.EXP1_TOTAL_STEPS
            self.action = None
            self.reward = 0
            
        elif self.exp_id == 2:
            self.env = HighStakesForaging(seed=cfg.SEED)
            self.worker = LocalRLWorker(self.env.observation_dim, self.env.action_dim)
            self.state = self.env.reset()
            self.step_i = 0
            self.total_steps = cfg.EXP2_TOTAL_STEPS
            self.action = None
            self.reward = 0
            self.died = False
            
        elif self.exp_id == 3:
            self.env = gym.make("CartPole-v1", render_mode="rgb_array")
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
            self.worker = LocalRLWorker(state_dim, action_dim)
            self.state, _ = self.env.reset(seed=cfg.SEED)
            self.episode_i = 0
            self.step_i = 0
            self.ep_reward = 0
            self.perturbed = False
            self.total_episodes = cfg.EXP3_TRAIN_EPISODES + cfg.EXP3_POST_EPISODES

    def run(self):
        running = True
        finished = False
        
        while running:
            dt = self.clock.tick(60)  # Maintain 60 FPS for smooth UI
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.speed = min(10000, int(self.speed * 1.5) + 1)
                    elif event.key == pygame.K_DOWN:
                        self.speed = max(1, int(self.speed / 1.5))
                        
            if not self.paused and not finished:
                self.step_accumulator += self.speed * (dt / 1000.0)
                steps_to_run = int(self.step_accumulator)
                self.step_accumulator -= steps_to_run
                
                for _ in range(steps_to_run):
                    if self.exp_id == 1:
                        finished = self._step_exp1()
                    elif self.exp_id == 2:
                        finished = self._step_exp2()
                    elif self.exp_id == 3:
                        finished = self._step_exp3()
                    
                    if finished:
                        break
            
            self.render(finished)
            
        pygame.quit()
        
    def _update_plots(self, modulation, reward):
        self.plot_hormones.add_data([modulation["DA_eff"], modulation["NA"], modulation["5HT"]])
        self.plot_hyperparams.add_data([modulation["alpha"], modulation["tau"], modulation["gamma"]])
        if self.exp_id in [1, 2]:
            self.plot_rewards.add_data([reward])

    # ---------------------------------------------------------
    # Experiment Logic Steps
    # ---------------------------------------------------------
    def _step_exp1(self):
        if self.step_i >= self.total_steps:
            return True
            
        hormone_signal = self.meta.engine.get_vector()[0]
        self.action = self.worker.select_action(self.state, hormone_signal=hormone_signal)
        next_state, self.reward, done, _, _ = self.env.step(self.action)
        
        self.worker.store_transition(self.state, self.action, self.reward, next_state, float(done))
        td_error = self.worker.update(hormone_signal=hormone_signal)
        modulation = self.meta.step(td_error, self.reward, done)
        self.worker.set_modulation(modulation["alpha"], modulation["tau"], modulation["gamma"])
        
        self.state = next_state
        self.step_i += 1
        self._update_plots(modulation, self.reward)
        return False

    def _step_exp2(self):
        if self.step_i >= self.total_steps:
            return True
            
        hormone_signal = self.meta.engine.get_vector()[0]
        self.action = self.worker.select_action(self.state, hormone_signal=hormone_signal)
        next_state, self.reward, done, _, info = self.env.step(self.action)
        self.died = info.get("death", False)
        
        self.worker.store_transition(self.state, self.action, self.reward, next_state, float(self.died))
        td_error = self.worker.update(hormone_signal=hormone_signal)
        modulation = self.meta.step(td_error, self.reward, self.died)
        self.worker.set_modulation(modulation["alpha"], modulation["tau"], modulation["gamma"])
        
        self.state = next_state
        self.step_i += 1
        
        if self.died:
            self.worker.reset_episode()
            
        self._update_plots(modulation, self.reward)
        return False

    def _step_exp3(self):
        if self.episode_i >= self.total_episodes:
            return True
            
        if self.episode_i == cfg.EXP3_PERTURB_EPISODE and not self.perturbed:
            self.env.unwrapped.gravity = cfg.EXP3_NEW_GRAVITY
            self.env.unwrapped.force_mag *= cfg.EXP3_NEW_FRICTION
            self.perturbed = True
            
        hormone_signal = self.meta.engine.get_vector()[0]
        action = self.worker.select_action(self.state, hormone_signal=hormone_signal)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        self.worker.store_transition(self.state, action, reward, next_state, float(done))
        td_error = self.worker.update(hormone_signal=hormone_signal)
        modulation = self.meta.step(td_error, reward, done)
        self.worker.set_modulation(modulation["alpha"], modulation["tau"], modulation["gamma"])
        
        self.ep_reward += reward
        self.state = next_state
        self.step_i += 1
        
        self._update_plots(modulation, reward)
        
        if done or self.step_i >= 500:
            self.plot_rewards.add_data([self.ep_reward])
            self.state, _ = self.env.reset()
            self.worker.reset_episode()
            self.episode_i += 1
            self.step_i = 0
            self.ep_reward = 0
            
        return False

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------
    def render(self, finished):
        self.screen.fill(C_BG)
        
        # Top Bar
        status = "FINISHED" if finished else ("PAUSED" if self.paused else "RUNNING")
        step_info = f"Step: {self.step_i}"
        if self.exp_id == 3:
            step_info = f"Episode: {self.episode_i} | Step: {self.step_i}"
            
        bar_text = f"Exp {self.exp_id} | {status} | Speed: {self.speed} steps/sec | {step_info}"
        bar_surf = self.font.render(bar_text, True, C_TEXT)
        self.screen.blit(bar_surf, (10, 10))
        
        controls = self.font.render("SPACE: Pause/Play | UP/DOWN: Adjust Speed", True, (150, 150, 150))
        self.screen.blit(controls, (self.width - controls.get_width() - 10, 10))
        
        # Story Environment
        pygame.draw.rect(self.screen, C_PANEL, self.story_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), self.story_rect, 1)
        
        if self.exp_id == 1:
            self._render_story_exp1()
        elif self.exp_id == 2:
            self._render_story_exp2()
        elif self.exp_id == 3:
            self._render_story_exp3()
            
        # Metrics
        self.plot_hormones.draw(self.screen)
        self.plot_hyperparams.draw(self.screen)
        self.plot_rewards.draw(self.screen)
        
        if finished:
            done_surf = self.huge_font.render("SIMULATION COMPLETE", True, (0, 255, 0))
            bg_rect = done_surf.get_rect(center=(self.width//2, self.height//2))
            pygame.draw.rect(self.screen, (0,0,0), bg_rect.inflate(40, 40))
            self.screen.blit(done_surf, bg_rect)
            
        pygame.display.flip()

    def _render_story_exp1(self):
        n_arms = cfg.EXP1_N_ARMS
        w = self.story_rect.width / n_arms
        
        for i in range(n_arms):
            bx = self.story_rect.x + i * w + 40
            by = self.story_rect.y + 60
            bw = w - 80
            bh = self.story_rect.height - 120
            
            # Highlight selected
            color = (60, 60, 80)
            if hasattr(self, 'action') and self.action == i:
                color = (180, 180, 80) # Yellowish pull
                
            pygame.draw.rect(self.screen, color, (bx, by, bw, bh))
            pygame.draw.rect(self.screen, (200, 200, 200), (bx, by, bw, bh), 2)
            
            # True Mean Bar
            means = self.env.means_pre if self.env._step < self.env.switch_step else self.env.means_post
            mu = means[i]
            bar_h = (mu / 12.0) * bh # max expected is 10
            bar_y = by + bh - bar_h
            if bar_h > 0:
                pygame.draw.rect(self.screen, (80, 200, 80), (bx+10, bar_y, bw-20, bar_h))
                
            lbl = self.large_font.render(f"Arm {i}", True, C_TEXT)
            self.screen.blit(lbl, (bx + bw/2 - lbl.get_width()/2, by - 35))
            
            mu_lbl = self.font.render(f"True μ={mu:.1f}", True, C_TEXT)
            self.screen.blit(mu_lbl, (bx + bw/2 - mu_lbl.get_width()/2, by + bh + 10))
            
        if self.env._step >= self.env.switch_step:
            alert = self.huge_font.render("ENVIRONMENT SWITCHED!", True, (255, 100, 100))
            self.screen.blit(alert, (self.story_rect.centerx - alert.get_width()/2, self.story_rect.y + 10))

    def _render_story_exp2(self):
        cx = self.story_rect.centerx
        cy = self.story_rect.centery
        
        safe_rect = pygame.Rect(self.story_rect.x + 150, cy - 80, 160, 160)
        risky_rect = pygame.Rect(self.story_rect.right - 310, cy - 80, 160, 160)
        
        pygame.draw.rect(self.screen, (80, 180, 80), safe_rect)
        pygame.draw.rect(self.screen, (180, 80, 80), risky_rect)
        
        safe_lbl = self.large_font.render("SAFE (+5)", True, (0, 0, 0))
        self.screen.blit(safe_lbl, (safe_rect.centerx - safe_lbl.get_width()/2, safe_rect.centery - 10))
        
        risky_lbl = self.large_font.render("RISKY (+50/DEATH)", True, (0, 0, 0))
        self.screen.blit(risky_lbl, (risky_rect.centerx - risky_lbl.get_width()/2, risky_rect.centery - 10))
        
        # Agent
        agent_x = cx
        agent_color = (150, 200, 255)
        
        if hasattr(self, 'action') and self.action is not None:
            if self.action == 0:
                agent_x -= 300
            else:
                agent_x += 300
                
        if hasattr(self, 'died') and self.died:
            agent_color = (255, 0, 0)
            pygame.draw.rect(self.screen, (255, 0, 0), self.story_rect, 5)
            skull = self.huge_font.render("DEATH (-500)", True, (255, 50, 50))
            self.screen.blit(skull, (agent_x - skull.get_width()/2, cy - 120))
        elif hasattr(self, 'reward'):
            rew = self.huge_font.render(f"+{self.reward}", True, (50, 255, 50) if self.reward > 0 else (255, 50, 50))
            self.screen.blit(rew, (agent_x - rew.get_width()/2, cy - 120))
            
        pygame.draw.circle(self.screen, agent_color, (int(agent_x), int(cy)), 40)
        pygame.draw.circle(self.screen, (255,255,255), (int(agent_x), int(cy)), 40, 3)

    def _render_story_exp3(self):
        try:
            frame = self.env.render()
            if frame is not None:
                # Resize keeping aspect ratio
                h, w, c = frame.shape
                aspect = w / h
                new_h = self.story_rect.height
                new_w = int(new_h * aspect)
                
                frame = np.transpose(frame, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                surf = pygame.transform.scale(surf, (new_w, new_h))
                
                # Center it
                offset_x = self.story_rect.x + (self.story_rect.width - new_w) // 2
                self.screen.blit(surf, (offset_x, self.story_rect.y))
        except Exception:
            pass
            
        if self.perturbed:
            alert = self.huge_font.render("GRAVITY x3 (PERTURBED)!", True, (255, 50, 50))
            self.screen.blit(alert, (self.story_rect.x + 20, self.story_rect.y + 20))

# ---------------------------------------------------------
# Fast Mode Runner
# ---------------------------------------------------------
def run_fast_mode(exp_id: int):
    print("=" * 60)
    print(f" Running Experiment {exp_id} in FAST Background Mode")
    print("=" * 60)
    
    start_time = time.time()
    if exp_id == 1:
        res = experiments.run_experiment_1(label="Fast Mode")
        evaluation.plot_experiment_1(res, suffix="_fast_mode")
    elif exp_id == 2:
        res = experiments.run_experiment_2(label="Fast Mode")
        evaluation.plot_experiment_2(res, suffix="_fast_mode")
    elif exp_id == 3:
        res = experiments.run_experiment_3(label="Fast Mode")
        evaluation.plot_experiment_3(res, suffix="_fast_mode")
        
    print(f"Completed in {time.time() - start_time:.2f} seconds.")
    print(f"Results dashboard saved to: {cfg.RESULTS_DIR}")

# ---------------------------------------------------------
# Full Ablation Runner
# ---------------------------------------------------------
def run_full_ablation_mode(exp_id: int):
    print("=" * 60)
    print(f" Running FULL ABLATION STUDY for Experiment {exp_id}")
    print("=" * 60)
    print(" This will run all 4 configurations (Full, Ablated NA, Ablated 5-HT, Static).")
    
    start_time = time.time()
    
    # Run ablation study for the specific experiment
    all_results = run_ablation_study(seed=cfg.SEED, exp_id=exp_id)
    
    # Generate comparative plots
    print("\n> Generating comparative bar charts...")
    plot_comparative_bars(all_results)
    
    # Export results
    print("\n> Exporting CSV results...")
    evaluation.export_csv(all_results)
    
    # Print conclusions
    print_scientific_conclusions(all_results)
    
    print(f"Full ablation study completed in {time.time() - start_time:.2f} seconds.")
    print(f"All dashboards and comparison plots saved to: {cfg.RESULTS_DIR}")

# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Neuromodulated RL Simulation Engine")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True,
                        help="Experiment to run: 1 (Bandit), 2 (Foraging), 3 (CartPole)")
    parser.add_argument("--mode", type=str, choices=["live", "fast", "ablation"], required=True,
                        help="Execution mode: live (UI), fast (background), or ablation (full analysis)")
    
    args = parser.parse_args()

    if args.mode == "fast":
        run_fast_mode(args.exp)
    elif args.mode == "ablation":
        run_full_ablation_mode(args.exp)
    else:
        print(f"\nStarting Live Simulation for Experiment {args.exp}...")
        print("Controls: [SPACE] Pause/Play | [UP/DOWN] Change Speed")
        engine = SimulationEngine(exp_id=args.exp)
        engine.run()
