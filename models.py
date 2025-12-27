import numpy as np
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Better for simple 1D data

class CountMinSketch:
    def __init__(self, width=2000, depth=5, seed=42):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.float32)
        rng = np.random.RandomState(seed)
        self.seeds = rng.randint(0, 2**31-1, size=depth)

    def _hash(self, key, i):
        return (int(key) ^ self.seeds[i]) % self.width

    def update(self, key, count=1.0):
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.table[i, idx] += count

    def query(self, key):
        vals = []
        for i in range(self.depth):
            idx = self._hash(key, i)
            vals.append(self.table[i, idx])
        return min(vals) if vals else 0.0

    def memory_bytes(self):
        return self.table.nbytes

class AdaSketch(CountMinSketch):
    def __init__(self, width=2000, depth=5, decay=0.9, decay_window=50, seed=42):
        super().__init__(width, depth, seed)
        self.decay = decay
        self.decay_window = decay_window
        self.t = 0

    def update(self, key, count=1.0):
        self.t += 1
        # Aggressive Decay: Forgets old history quickly
        if self.t % self.decay_window == 0:
            self.table *= self.decay
        super().update(key, count)

class SlidingWindowCBF:
    def __init__(self, width=4000, num_hashes=4, window_size=2000, seed=42):
        self.width = width
        self.window_size = window_size
        self.window = deque()
        self.counters = np.zeros(width, dtype=int)
        rng = np.random.RandomState(seed)
        self.seeds = rng.randint(0, 2**31-1, size=num_hashes)

    def _hashes(self, key):
        return [(int(key) ^ s) % self.width for s in self.seeds]

    def add(self, key):
        self.window.append(key)
        for h in self._hashes(key):
            self.counters[h] += 1
        if len(self.window) > self.window_size:
            old = self.window.popleft()
            for h in self._hashes(old):
                if self.counters[h] > 0:
                    self.counters[h] -= 1

    def contains(self, key):
        return all(self.counters[h] > 0 for h in self._hashes(key))
    
    def memory_bytes(self):
        return self.counters.nbytes

class StableLearnedBloomFilter:
    def __init__(self, width=2000, window_size=2000, retrain_every=200):
        self.backup = SlidingWindowCBF(width=width, window_size=window_size)
        # Decision Tree works better for sharp "Jumps" in data than Logistic Regression
        self.clf = DecisionTreeClassifier(max_depth=5) 
        self.retrain_every = retrain_every
        self.X_train = []
        self.y_train = []
        self.age = 0
        self.fitted = False

    def check_and_add(self, key, feature_val):
        is_normal = False
        
        # 1. Classifier Check (Lower threshold to 0.6 to trust it more)
        if self.fitted:
            try:
                prob = self.clf.predict_proba([[feature_val]])[0, 1]
                if prob > 0.6: 
                    is_normal = True
            except:
                pass
        
        # 2. Backup Check
        if not is_normal:
            if self.backup.contains(key):
                is_normal = True
        
        # 3. Add to structures
        self.backup.add(key)
        self.age += 1
        
        # Train Data Collection
        self.X_train.append([feature_val])
        self.y_train.append(1) # Normal
        
        # Synthetic Anomaly (Contrastive Noise)
        fake_val = feature_val + np.random.choice([-50, 50, 100])
        self.X_train.append([fake_val])
        self.y_train.append(0) # Anomaly
        
        if self.age % self.retrain_every == 0:
            self._retrain()
            
        return not is_normal 

    def _retrain(self):
        # Shorter history for faster adaptation
        window = 500
        if len(self.X_train) > window:
            self.X_train = self.X_train[-window:]
            self.y_train = self.y_train[-window:]
        try:
            self.clf.fit(self.X_train, self.y_train)
            self.fitted = True
        except:
            pass

    def memory_bytes(self):
        return self.backup.memory_bytes()