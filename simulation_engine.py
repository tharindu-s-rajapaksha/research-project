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
from worker import LocalRLWorker, StaticBaselineWorker
from environments import VolatileBandit, HighStakesForaging
import experiments
import evaluation
from ablation import (run_multiseed_study, plot_comparative_bars_multiseed,
                      print_scientific_conclusions_multiseed)

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

# UI Window Settings
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900

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
        self.total_points = 0
        self.font = pygame.font.SysFont("Consolas", 12)
        self.title_font = pygame.font.SysFont("Consolas", 14, bold=True)
        
    def add_data(self, vals):
        self.total_points += 1
        for i, v in enumerate(vals):
            self.data[i].append(v)
            
    def clear(self):
        self.total_points = 0
        for q in self.data:
            q.clear()
            
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
                val_str = f": {self.data[i][-1]:.4f}"
            
            l_surf = self.font.render(f"{label}{val_str}", True, C_TEXT)
            surface.blit(l_surf, (lx + 15, ly))
            lx += l_surf.get_width() + 25
            
        # Determine global Y bounds first
        all_vals = [v for dq in self.data for v in dq]
        min_y = min(self.y_range[0], min(all_vals) if all_vals else self.y_range[0])
        max_y = max(self.y_range[1], max(all_vals) if all_vals else self.y_range[1])
        if min_y == max_y: max_y = min_y + 1

        # Draw Horizontal Grids (Y-axis) with Values
        grid_color = (70, 70, 80)
        n_grids = 4
        bottom_margin = 18
        top_margin = 45
        chart_height = self.rect.height - bottom_margin - top_margin
        
        for k in range(n_grids):
            val = min_y + (max_y - min_y) * (k / (n_grids - 1))
            norm = (val - min_y) / (max_y - min_y + 1e-6)
            py = self.rect.y + self.rect.height - bottom_margin - norm * chart_height
            
            # Draw line
            pygame.draw.line(surface, grid_color, (self.rect.x, py), (self.rect.x + self.rect.width, py), 1)
            
            # Draw text label on the right side
            val_surf = self.font.render(f"{val:.1f}", True, (150, 150, 150))
            surface.blit(val_surf, (self.rect.x + self.rect.width - val_surf.get_width() - 5, py - 14))

        # Draw Vertical Grids (X-axis) with Values
        x_step = 50
        start_x = max(0, self.total_points - self.max_pts)
        end_x = self.total_points
        first_grid = (start_x // x_step) * x_step
        if first_grid < start_x:
            first_grid += x_step
            
        for v_step in range(first_grid, end_x + 1, x_step):
            j = v_step - start_x
            px = self.rect.x + (j / max(1, (self.max_pts - 1))) * self.rect.width
            if self.rect.x <= px <= self.rect.x + self.rect.width:
                pygame.draw.line(surface, grid_color, (px, self.rect.y + top_margin), (px, self.rect.y + self.rect.height - bottom_margin), 1)
                val_surf = self.font.render(f"{v_step}", True, (200, 200, 200))
                # Center text over the vertical line
                surface.blit(val_surf, (px - val_surf.get_width() / 2, self.rect.y + self.rect.height - bottom_margin + 2))

        # Lines
        for i, q in enumerate(self.data):
            if len(q) < 2: continue
            pts = []
            for j, val in enumerate(q):
                px = self.rect.x + (j / (self.max_pts - 1)) * self.rect.width
                val = max(min_y, min(max_y, val))
                norm = (val - min_y) / (max_y - min_y + 1e-6)
                py = self.rect.y + self.rect.height - bottom_margin - norm * chart_height
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
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Experiment {exp_id} - Live Simulation")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("Consolas", 14)
        self.large_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.huge_font = pygame.font.SysFont("Consolas", 36, bold=True)
        
        # Dashboard Layout (Relative to window size)
        margin = 10
        story_h = int(self.height * 0.32)
        story_w = int(self.width * 0.78) - margin
        self.story_rect = pygame.Rect(margin, 40, story_w, story_h)
        self.metrics_rect = pygame.Rect(story_w + 2*margin, 40, self.width - story_w - 3*margin, story_h)
        
        # Calculate row dimensions
        charts_start_y = story_h + 50
        available_charts_h = self.height - charts_start_y - margin
        row_h = available_charts_h // 3
        chart_h = row_h - margin
        chart_w = (self.width - 4*margin) // 3
        
        # Row 1: Individual Hormone Charts
        y1 = charts_start_y
        self.plot_da = ScrollingPlot(margin, y1, chart_w, chart_h, "Dopamine (DA_eff)", ["DA_eff"], [C_DA], y_range=(0, 1))
        self.plot_na = ScrollingPlot(2*margin + chart_w, y1, chart_w, chart_h, "Noradrenaline (NA)", ["NA"], [C_NA], y_range=(0, 1))
        self.plot_5ht = ScrollingPlot(3*margin + 2*chart_w, y1, chart_w, chart_h, "Serotonin (5-HT)", ["5HT"], [C_5HT], y_range=(0, 1))
        
        # Row 2: Individual Hyperparameter Charts
        y2 = y1 + row_h
        self.plot_alpha = ScrollingPlot(margin, y2, chart_w, chart_h, "Alpha (Learning Rate)", ["Alpha"], [C_ALPHA], y_range=(0, 0.005))
        self.plot_tau = ScrollingPlot(2*margin + chart_w, y2, chart_w, chart_h, "Epsilon (Explore Rate)", ["Epsilon"], [C_TAU], y_range=(0, 1))
        self.plot_gamma = ScrollingPlot(3*margin + 2*chart_w, y2, chart_w, chart_h, "Gamma (Discount)", ["Gamma"], [C_GAMMA], y_range=(0, 1))
        
        # Row 3: Instantaneous Reward (Full Width)
        y3 = y2 + row_h
        self.plot_rewards = ScrollingPlot(margin, y3, self.width - 2*margin, chart_h, "Instantaneous Reward", ["Reward"], [(50, 150, 255)], y_range=(-10, 20))
        
        self.setup_experiment()
        
    def setup_experiment(self):
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        
        self.step_accumulator = 0.0
        self.plot_da.clear()
        self.plot_na.clear()
        self.plot_5ht.clear()
        self.plot_alpha.clear()
        self.plot_tau.clear()
        self.plot_gamma.clear()
        self.plot_rewards.clear()
        
        if not hasattr(self, 'ablation_flags'):
            self.ablation_flags = {"DA": True, "NA": True, "5HT": True}
            
        self.meta = HormonalMetaAgent(
            enable_da=self.ablation_flags["DA"],
            enable_na=self.ablation_flags["NA"],
            enable_5ht=self.ablation_flags["5HT"],
        )
        
        is_static = not self.ablation_flags["DA"] and not self.ablation_flags["NA"] and not self.ablation_flags["5HT"]
        
        if self.exp_id == 1:
            self.env = VolatileBandit(seed=cfg.SEED)
            if is_static:
                self.worker = StaticBaselineWorker(self.env.observation_dim, self.env.action_dim)
            else:
                self.worker = LocalRLWorker(self.env.observation_dim, self.env.action_dim)
            self.state = self.env.reset()
            self.step_i = 0
            self.total_steps = cfg.EXP1_TOTAL_STEPS
            self.action = None
            self.reward = 0
            # Live Metrics
            self.cum_reward = 0.0
            self.optimal_count = 0
            self.optimal_percentage = 0.0
            self.switched_timer = 0
            
        elif self.exp_id == 2:
            self.env = HighStakesForaging(seed=cfg.SEED)
            if is_static:
                self.worker = StaticBaselineWorker(self.env.observation_dim, self.env.action_dim)
            else:
                self.worker = LocalRLWorker(self.env.observation_dim, self.env.action_dim)
            self.state = self.env.reset()
            self.step_i = 0
            self.total_steps = cfg.EXP2_TOTAL_STEPS
            self.action = None
            self.reward = 0
            self.died = False
            # Live Metrics
            self.cum_reward = 0.0
            self.death_count = 0
            self.survival_steps = 0
            
        elif self.exp_id == 3:
            self.env = gym.make("CartPole-v1", render_mode="rgb_array")
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
            if is_static:
                self.worker = StaticBaselineWorker(state_dim, action_dim)
            else:
                self.worker = LocalRLWorker(state_dim, action_dim)
            self.state, _ = self.env.reset(seed=cfg.SEED)
            self.episode_i = 0
            self.step_i = 0
            self.ep_reward = 0
            self.perturbed = False
            # Live Metrics
            self.cum_reward = 0.0
            self.success_count = 0
            self.last_ep_score = 0.0
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
                    elif event.key == pygame.K_r:
                        self.setup_experiment()
                        finished = False
                    elif event.key == pygame.K_q:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if hasattr(self, 'restart_rect') and self.restart_rect.collidepoint(event.pos):
                            self.setup_experiment()
                            finished = False
                        if hasattr(self, 'checkboxes'):
                            for name, rect in self.checkboxes.items():
                                if rect.collidepoint(event.pos):
                                    self.ablation_flags[name] = not self.ablation_flags[name]
                                    if name == "DA":
                                        self.meta.engine.enable_da = self.ablation_flags["DA"]
                                    elif name == "NA":
                                        self.meta.engine.enable_na = self.ablation_flags["NA"]
                                    elif name == "5HT":
                                        self.meta.engine.enable_5ht = self.ablation_flags["5HT"]
                        
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
        self.plot_da.add_data([modulation["DA_eff"]])
        self.plot_na.add_data([modulation["NA"]])
        self.plot_5ht.add_data([modulation["5HT"]])
        
        self.plot_alpha.add_data([modulation["alpha"]])
        self.plot_tau.add_data([modulation["epsilon"]])
        self.plot_gamma.add_data([modulation["gamma"]])
        
        if self.exp_id in [1, 2]:
            self.plot_rewards.add_data([reward])

    # ---------------------------------------------------------
    # Experiment Logic Steps
    # ---------------------------------------------------------
    def _step_exp1(self):
        if self.step_i >= self.total_steps:
            return True
            
        hormone_signal = self.meta.engine.plastic_gate()
        self.action = self.worker.select_action(self.state, hormone_signal=hormone_signal)
        next_state, self.reward, done, _, info = self.env.step(self.action)
        
        self.worker.store_transition(self.state, self.action, self.reward, next_state, float(done))
        td_error = self.worker.update(hormone_signal=hormone_signal)
        modulation = self.meta.step(td_error, self.reward, done)
        self.worker.set_modulation(modulation["alpha"], modulation["epsilon"], modulation["gamma"], modulation["punish_gain"])
        
        # Update metrics
        self.cum_reward += self.reward
        if info["optimal_arm"] == self.action:
            self.optimal_count += 1
        self.optimal_percentage = (self.optimal_count / (self.step_i + 1)) * 100
        
        if info["switched"]:
            self.switched_timer = 20  # Show label for 20 frames/steps
        elif self.switched_timer > 0:
            self.switched_timer -= 1
            
        # ===== Debugging Print =====

        # Chart 1: Hormones
        # print(f"step {self.step_i}: reward={self.reward}, td_error={td_error}, da={modulation['DA']}, na={modulation['NA']}, ht={modulation['5HT']}")

        # Chart 2: Hyperparameters
        # print(f"step {self.step_i} | DA_eff: {modulation['DA_eff']:.2f}, NA: {modulation['NA']:.2f}, 5HT: {modulation['5HT']:.2f} | Alpha: {modulation['alpha']:.4f}, Tau: {modulation['tau']:.2f}, Gamma: {modulation['gamma']:.2f}")

        self.state = next_state
        self.step_i += 1
        self._update_plots(modulation, self.reward)
        return False

    def _step_exp2(self):
        if self.step_i >= self.total_steps:
            return True
            
        hormone_signal = self.meta.engine.plastic_gate()
        self.action = self.worker.select_action(self.state, hormone_signal=hormone_signal)
        next_state, self.reward, done, _, info = self.env.step(self.action)
        self.died = info.get("death", False)
        
        self.worker.store_transition(self.state, self.action, self.reward, next_state, float(self.died))
        td_error = self.worker.update(hormone_signal=hormone_signal)
        modulation = self.meta.step(td_error, self.reward, self.died)
        self.worker.set_modulation(modulation["alpha"], modulation["epsilon"], modulation["gamma"], modulation["punish_gain"])
        
        # Update metrics
        self.cum_reward += self.reward
        self.survival_steps += 1
        if self.died:
            self.death_count += 1
            self.survival_steps = 0
            
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
            self.env.unwrapped.force_mag *= cfg.EXP3_FORCE_SCALE
            self.perturbed = True
            
        hormone_signal = self.meta.engine.plastic_gate()
        action = self.worker.select_action(self.state, hormone_signal=hormone_signal)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        self.worker.store_transition(self.state, action, reward, next_state, float(done))
        td_error = self.worker.update(hormone_signal=hormone_signal)
        modulation = self.meta.step(td_error, reward, done)
        self.worker.set_modulation(modulation["alpha"], modulation["epsilon"], modulation["gamma"], modulation["punish_gain"])
        
        self.ep_reward += reward
        self.cum_reward += reward
        self.state = next_state
        self.step_i += 1
        
        self._update_plots(modulation, reward)
        
        if done or self.step_i >= 500:
            self.last_ep_score = self.ep_reward
            if self.ep_reward >= cfg.EXP3_RECOVERY_TARGET:  # CartPole success bar
                self.success_count += 1
                
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
            
        bar_text = f"Exp {self.exp_id} | {status} | Speed: {self.speed} steps/sec | {step_info} | MODEL: "
        bar_surf = self.font.render(bar_text, True, C_TEXT)
        self.screen.blit(bar_surf, (10, 10))
        
        if isinstance(self.worker, StaticBaselineWorker):
            model_surf = self.font.render("STATIC BASELINE", True, (255, 100, 100))
        else:
            model_surf = self.font.render("DYNAMIC RL", True, (100, 255, 100))
        self.screen.blit(model_surf, (10 + bar_surf.get_width(), 10))
        
        controls = self.font.render("SPACE: Pause/Play | UP/DOWN: Adjust Speed | Q: Exit", True, (150, 150, 150))
        self.screen.blit(controls, (self.width - controls.get_width() - 10, 15))
        
        # Restart Button
        self.restart_rect = pygame.Rect(self.width - controls.get_width() - 140, 10, 120, 25)
        pygame.draw.rect(self.screen, (200, 50, 50), self.restart_rect)
        btn_text = self.font.render("RESTART (R)", True, (255, 255, 255))
        self.screen.blit(btn_text, (self.restart_rect.centerx - btn_text.get_width()/2, self.restart_rect.centery - btn_text.get_height()/2))
        
        # Ablation Checkboxes
        cb_x = 750
        lbl_ablation = self.font.render("Ablation:", True, C_TEXT)
        self.screen.blit(lbl_ablation, (cb_x, 15))
        cb_x += lbl_ablation.get_width() + 10
        
        self.checkboxes = {}
        for name in ["DA", "NA", "5HT"]:
            rect = pygame.Rect(cb_x, 15, 15, 15)
            self.checkboxes[name] = rect
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2 if not self.ablation_flags[name] else 0)
            if self.ablation_flags[name]:
                pygame.draw.rect(self.screen, (100, 255, 100), rect)
            
            lbl = self.font.render(name, True, C_TEXT)
            self.screen.blit(lbl, (cb_x + 20, 15))
            cb_x += 60
        
        # Story Environment
        pygame.draw.rect(self.screen, C_PANEL, self.story_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), self.story_rect, 1)
        
        if self.exp_id == 1:
            self._render_story_exp1()
        elif self.exp_id == 2:
            self._render_story_exp2()
        elif self.exp_id == 3:
            self._render_story_exp3()
            
        # Metrics Panel (Right Side)
        self._render_metrics()
            
        # Charts
        self.plot_da.draw(self.screen)
        self.plot_na.draw(self.screen)
        self.plot_5ht.draw(self.screen)
        
        self.plot_alpha.draw(self.screen)
        self.plot_tau.draw(self.screen)
        self.plot_gamma.draw(self.screen)
        
        self.plot_rewards.draw(self.screen)
        
        if finished:
            done_surf = self.huge_font.render("SIMULATION COMPLETE", True, (0, 255, 0))
            bg_rect = done_surf.get_rect(center=(self.width//2, self.height//2))
            pygame.draw.rect(self.screen, (0,0,0), bg_rect.inflate(40, 40))
            self.screen.blit(done_surf, bg_rect)
            
        pygame.display.flip()

    def _render_metrics(self):
        # Draw Metrics Panel
        pygame.draw.rect(self.screen, C_PANEL, self.metrics_rect)
        pygame.draw.rect(self.screen, (150, 150, 150), self.metrics_rect, 1)
        
        title = self.large_font.render("LIVE EVALUATION", True, (255, 200, 50))
        self.screen.blit(title, (self.metrics_rect.x + 20, self.metrics_rect.y + 15))

        if self.exp_id == 1:
            phase = sum(1 for s in cfg.EXP1_SWITCH_STEPS if self.env._step >= s)
            if phase > 0:
                sw_lbl = self.large_font.render(f"PHASE {phase} !", True, (255, 50, 50))
                self.screen.blit(sw_lbl, (self.metrics_rect.x + 20, self.metrics_rect.y + 40))
        
        y_off = self.metrics_rect.y + 80
        spacing = 35
        
        metrics = []
        if self.exp_id == 1:
            metrics = [
                ("Total Reward", f"{self.cum_reward:.1f}"),
                ("Optimal Pulls", f"{self.optimal_count}"),
                ("Optimal %", f"{self.optimal_percentage:.1f}%"),
            ]
        elif self.exp_id == 2:
            metrics = [
                ("Total Reward", f"{self.cum_reward:.1f}"),
                ("Death Count", f"{self.death_count}"),
                ("Survival Streak", f"{self.survival_steps} steps"),
                ("Risk Level", "HIGH" if self.action == 1 else "LOW")
            ]
        elif self.exp_id == 3:
            metrics = [
                ("Total Score", f"{self.cum_reward:.1f}"),
                ("Success Count", f"{self.success_count}"),
                ("Last Ep Score", f"{self.last_ep_score:.1f}"),
                ("Status", "PERTURBED" if self.perturbed else "NORMAL")
            ]
            
        for label, val in metrics:
            lbl_surf = self.font.render(label + ":", True, (200, 200, 200))
            val_surf = self.font.render(val, True, (255, 255, 255))
            self.screen.blit(lbl_surf, (self.metrics_rect.x + 20, y_off))
            self.screen.blit(val_surf, (self.metrics_rect.right - val_surf.get_width() - 20, y_off))
            y_off += spacing

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
            mu = self.env.current_means[i]
            bar_h = (mu / 12.0) * bh # max expected is 10
            bar_y = by + bh - bar_h
            if bar_h > 0:
                pygame.draw.rect(self.screen, (80, 200, 80), (bx+10, bar_y, bw-20, bar_h))
                
            lbl = self.large_font.render(f"Arm {i}", True, C_TEXT)
            self.screen.blit(lbl, (bx + bw/2 - lbl.get_width()/2, by - 35))
            
            mu_lbl = self.font.render(f"True μ={mu:.1f}", True, C_TEXT)
            self.screen.blit(mu_lbl, (bx + bw/2 - mu_lbl.get_width()/2, by + bh + 10))

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
    print(f" Configs: {list(cfg.ABLATION_CONFIGS.keys())}")
    print(f" Seeds:   {cfg.SEEDS}")

    start_time = time.time()

    # Run the multi-seed study for the specific experiment
    all_ms = run_multiseed_study(seeds=cfg.SEEDS, exp_id=exp_id)

    # Dashboards from the first seed (representative visual)
    plotters = {1: evaluation.plot_experiment_1, 2: evaluation.plot_experiment_2,
                3: evaluation.plot_experiment_3}
    for label, res_list in all_ms[f"Experiment_{exp_id}"].items():
        if res_list:
            sfx = f"_{label.replace(' ', '_').lower()}"
            plotters[exp_id](res_list[0], suffix=sfx)

    # Comparative plots + statistics
    print("\n> Generating comparative bar charts (mean ± 95% CI)...")
    plot_comparative_bars_multiseed(all_ms)
    print("\n> Summarizing metrics + significance tests...")
    evaluation.summarize_multiseed(all_ms)
    evaluation.compute_multiseed_pvalues(all_ms)
    print_scientific_conclusions_multiseed(all_ms)

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
