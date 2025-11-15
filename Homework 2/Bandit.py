"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        pass

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    pass

#--------------------------------------#

class ThompsonSampling(Bandit):
    pass




def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")


"""
HW2 — Multi-Armed Bandits (simplified, documented)

Implements:
    - Epsilon-Greedy (epsilon = 1/t)
    - Thompson Sampling (Gaussian, known precision)

Saves:
    - experiment_results.csv  (Bandit, Reward, Algorithm)
    - plot1_learning_linear.png
    - plot1_learning_logx.png
    - plot2_cumulative_rewards.png
    - plot2_cumulative_regrets.png

Logging:
    Uses loguru for logging (instead of print).

Usage:
    $ python bandit_hw2.py

Notes:
    Docstrings follow Google style so they can be parsed and reformatted by `pyment`.
"""

from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

# ----- config from PDF -----
BANDIT_REWARD = [1, 2, 3, 4]     # true means
N_TRIALS = 20_000
TAU = 1.0                        # known precision for Gaussian rewards (variance = 1/TAU)

# ----- base class (do not remove anything) -----
class Bandit(ABC):
    """Abstract base class for bandit algorithms."""

    @abstractmethod
    def __init__(self, p):
        """Initialize the algorithm with arm parameters.

        Args:
            p (list[float]): True means (or parameters) for each arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return human-readable short description of the instance.

        Returns:
            str: Short representation used in logs.
        """
        pass

    @abstractmethod
    def pull(self):
        """Select an arm and obtain a reward.

        Returns:
            tuple[int, float]: (arm_index, observed_reward).
        """
        pass

    @abstractmethod
    def update(self):
        """Update internal state after observing a reward.

        Notes:
            Concrete subclasses should define signature:
                update(arm: int, reward: float) -> None
        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the algorithm for a number of trials.

        Notes:
            Concrete subclasses should define signature:
                experiment(n: int = N_TRIALS) -> None
        """
        pass

    @abstractmethod
    def report(self):
        """Summarize results and return per-step records.

        Returns:
            pandas.DataFrame: Columns {Bandit, Reward, Algorithm}.
        """
        pass

# ----- tiny helpers -----
def gauss_reward(mean, tau=TAU):
    """Sample a Gaussian reward with known precision.

    Args:
        mean (float): True mean of the selected arm.
        tau (float): Known precision (1/variance). Defaults to TAU.

    Returns:
        float: Sampled reward from N(mean, 1/tau).
    """
    std = (1.0 / tau) ** 0.5
    return random.gauss(mean, std)

def cumulative_avg(x):
    """Compute cumulative average of a 1-D sequence.

    Args:
        x (array-like): Sequence of numeric values.

    Returns:
        numpy.ndarray: y[i] = mean(x[: i+1]).
    """
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / (np.arange(len(x)) + 1)

# ----- Visualization -----
class Visualization:
    """Minimal plotting helper for learning/reward/regret curves."""

    def __init__(self, histories):
        """Construct the visualization helper.

        Args:
            histories (dict[str, dict[str, list]]): Mapping
                algo_name -> {
                    "rewards": list[float],
                    "cum_rewards": list[float],
                    "cum_regrets": list[float]
                }.
        """
        self.histories = histories  # dict: name -> dict with 'cum_rewards', 'cum_regrets'

    def plot1(self):
        """Plot learning curves (cumulative average reward) with linear and log-x scales.

        Saves:
            - plot1_learning_linear.png
            - plot1_learning_logx.png
        """
        # learning curves: cumulative average reward (linear and log x)
        for logx, path, title in [
            (False, "plot1_learning_linear.png", "Learning curve (linear)"),
            (True,  "plot1_learning_logx.png", "Learning curve (log x)"),
        ]:
            plt.figure()
            for algo_name, h in self.histories.items():
                avg = cumulative_avg(h["rewards"])
                x = np.arange(1, len(avg) + 1)
                (plt.semilogx if logx else plt.plot)(x, avg, label=algo_name)
            plt.xlabel("Trial (log scale)" if logx else "Trial")
            plt.ylabel("Cumulative average reward")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

    def plot2(self):
        """Plot cumulative rewards and cumulative regrets.

        Saves:
            - plot2_cumulative_rewards.png
            - plot2_cumulative_regrets.png
        """
        # cumulative rewards
        plt.figure()
        for algo_name, h in self.histories.items():
            x = np.arange(1, len(h["cum_rewards"]) + 1)
            plt.plot(x, h["cum_rewards"], label=algo_name)
        plt.xlabel("Trial"); plt.ylabel("Cumulative reward")
        plt.title("Cumulative rewards: ε-Greedy vs Thompson")
        plt.legend(); plt.tight_layout()
        plt.savefig("plot2_cumulative_rewards.png", dpi=150); plt.close()

        # cumulative regrets
        plt.figure()
        for algo_name, h in self.histories.items():
            x = np.arange(1, len(h["cum_regrets"]) + 1)
            plt.plot(x, h["cum_regrets"], label=algo_name)
        plt.xlabel("Trial"); plt.ylabel("Cumulative regret")
        plt.title("Cumulative regrets: ε-Greedy vs Thompson")
        plt.legend(); plt.tight_layout()
        plt.savefig("plot2_cumulative_regrets.png", dpi=150); plt.close()

# ----- Epsilon-Greedy -----
class EpsilonGreedy(Bandit):
    """Epsilon-Greedy policy with ε(t) = min(1, ε0 / t)."""

    def __init__(self, p, eps0=1.0):
        """Initialize Epsilon-Greedy learner.

        Args:
            p (list[float]): True means for each arm.
            eps0 (float): Initial epsilon constant (default 1.0).
        """
        self.means = list(p)
        self.k = len(self.means)
        self.counts = np.zeros(self.k, dtype=int)
        self.est = np.zeros(self.k, dtype=float)
        self.eps0 = eps0
        self.opt = max(self.means)
        self.name = "Epsilon-Greedy"

        self.chosen, self.rewards = [], []
        self.cum_rewards, self.cum_regrets = [], []

    def __repr__(self):
        """Short string representation for logging/debugging.

        Returns:
            str: Configuration summary.
        """
        return f"{self.name}(eps0={self.eps0})"

    def _eps(self, t):
        """Compute ε at time step t.

        Args:
            t (int): 1-based trial index.

        Returns:
            float: Exploration probability at step t.
        """
        return min(1.0, self.eps0 / max(1, t))

    def _select(self, t):
        """Select an arm via ε-greedy rule.

        Args:
            t (int): 1-based trial index.

        Returns:
            int: Selected arm index.
        """
        return random.randrange(self.k) if random.random() < self._eps(t) else int(np.argmax(self.est))

    def pull(self):
        """Not used directly; selection occurs in experiment().

        Raises:
            RuntimeError: Always, to nudge usage of experiment().
        """
        raise RuntimeError("Use experiment()")

    def update(self, arm, r):
        """Update running mean estimate for a pulled arm.

        Args:
            arm (int): Index of the pulled arm.
            r (float): Observed reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.est[arm] += (r - self.est[arm]) / n

    def experiment(self, n=N_TRIALS):
        """Run ε-greedy for n steps and record metrics.

        Args:
            n (int, optional): Number of trials. Defaults to N_TRIALS.
        """
        cr = 0.0; cg = 0.0
        for t in range(1, n + 1):
            a = self._select(t)
            r = gauss_reward(self.means[a], TAU)
            self.update(a, r)

            cr += r
            cg += (self.opt - self.means[a])

            self.chosen.append(a)
            self.rewards.append(r)
            self.cum_rewards.append(cr)
            self.cum_regrets.append(cg)

    def report(self):
        """Log summary and return per-step records.

        Returns:
            pandas.DataFrame: Columns {Bandit, Reward, Algorithm}.
        """
        total_r = self.cum_rewards[-1]
        total_g = self.cum_regrets[-1]
        logger.info(f"[{self.name}] cumulative reward = {total_r:.4f}")
        logger.info(f"[{self.name}] cumulative regret = {total_g:.4f}")
        logger.debug(f"[{self.name}] avg reward = {total_r/len(self.rewards):.4f} | "
                     f"avg regret = {total_g/len(self.rewards):.4f}")

        return pd.DataFrame({
            "Bandit": self.chosen,
            "Reward": [float(x) for x in self.rewards],
            "Algorithm": [self.name] * len(self.rewards)
        })

# ----- Thompson Sampling (Gaussian, known precision) -----
class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision."""

    def __init__(self, p, mu0=0.0, tau0=1e-6, tau=TAU):
        """Initialize Thompson Sampling learner.

        Args:
            p (list[float]): True means for each arm.
            mu0 (float): Prior mean of arm means. Defaults to 0.0.
            tau0 (float): Prior precision of arm means. Defaults to 1e-6.
            tau (float): Known observation precision. Defaults to TAU.
        """
        self.means = list(p)
        self.k = len(self.means)
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau
        self.counts = np.zeros(self.k, dtype=int)
        self.sumr = np.zeros(self.k, dtype=float)
        self.opt = max(self.means)
        self.name = "Thompson"

        self.chosen, self.rewards = [], []
        self.cum_rewards, self.cum_regrets = [], []

    def __repr__(self):
        """Short string representation for logging/debugging.

        Returns:
            str: Configuration summary.
        """
        return f"{self.name}(tau={self.tau}, mu0={self.mu0}, tau0={self.tau0})"

    def _post(self, a):
        """Compute posterior parameters for arm a (Normal-Normal).

        Args:
            a (int): Arm index.

        Returns:
            tuple[float, float]: (posterior_mean, posterior_precision).
        """
        post_tau = self.tau0 + self.counts[a] * self.tau
        post_mu = (self.tau0 * self.mu0 + self.tau * self.sumr[a]) / post_tau
        return post_mu, post_tau

    def _select(self):
        """Sample from each arm posterior, pick the max sample.

        Returns:
            int: Selected arm index.
        """
        samples = []
        for a in range(self.k):
            m, t = self._post(a)
            std = (1.0 / t) ** 0.5
            samples.append(random.gauss(m, std))
        return int(np.argmax(samples))

    def pull(self):
        """Not used directly; selection occurs in experiment().

        Raises:
            RuntimeError: Always, to nudge usage of experiment().
        """
        raise RuntimeError("Use experiment()")

    def update(self, a, r):
        """Update sufficient statistics (count and sum of rewards).

        Args:
            a (int): Index of the pulled arm.
            r (float): Observed reward.
        """
        self.counts[a] += 1
        self.sumr[a] += r

    def experiment(self, n=N_TRIALS):
        """Run Thompson Sampling for n steps and record metrics.

        Args:
            n (int, optional): Number of trials. Defaults to N_TRIALS.
        """
        cr = 0.0; cg = 0.0
        for _ in range(n):
            a = self._select()
            r = gauss_reward(self.means[a], self.tau)
            self.update(a, r)

            cr += r
            cg += (self.opt - self.means[a])

            self.chosen.append(a)
            self.rewards.append(r)
            self.cum_rewards.append(cr)
            self.cum_regrets.append(cg)

    def report(self):
        """Log summary and return per-step records.

        Returns:
            pandas.DataFrame: Columns {Bandit, Reward, Algorithm}.
        """
        total_r = self.cum_rewards[-1]
        total_g = self.cum_regrets[-1]
        logger.info(f"[{self.name}] cumulative reward = {total_r:.4f}")
        logger.info(f"[{self.name}] cumulative regret = {total_g:.4f}")
        logger.debug(f"[{self.name}] avg reward = {total_r/len(self.rewards):.4f} | "
                     f"avg regret = {total_g/len(self.rewards):.4f}")

        return pd.DataFrame({
            "Bandit": self.chosen,
            "Reward": [float(x) for x in self.rewards],
            "Algorithm": [self.name] * len(self.rewards)
        })

# ----- simple visual comparison -----
def comparison(histories):
    """Create and save all plots required by the assignment.

    Args:
        histories (dict[str, dict[str, list]]): Data recorded for each algorithm.
    """
    viz = Visualization(histories)
    viz.plot1()
    viz.plot2()

# ----- main -----
if __name__ == '__main__':
    """Entry point to run both algorithms and produce artifacts."""
    # replace prints with logging
    logger.info("Starting HW2 experiments (ε-Greedy vs Thompson)")

    # optional: set seeds if you want repeatable runs
    # random.seed(0); np.random.seed(0)

    eg = EpsilonGreedy(BANDIT_REWARD, eps0=1.0)
    ts = ThompsonSampling(BANDIT_REWARD, mu0=0.0, tau0=1e-6, tau=TAU)

    eg.experiment(N_TRIALS)
    ts.experiment(N_TRIALS)

    df = pd.concat([eg.report(), ts.report()], ignore_index=True)
    df.to_csv("experiment_results.csv", index=False)
    logger.info("Saved CSV: experiment_results.csv")

    histories = {
        "Epsilon-Greedy": {
            "rewards": eg.rewards,
            "cum_rewards": eg.cum_rewards,
            "cum_regrets": eg.cum_regrets
        },
        "Thompson": {
            "rewards": ts.rewards,
            "cum_rewards": ts.cum_rewards,
            "cum_regrets": ts.cum_regrets
        }
    }
    comparison(histories)

    logger.info("Saved plots: plot1_learning_linear.png, plot1_learning_logx.png, "
                "plot2_cumulative_rewards.png, plot2_cumulative_regrets.png")