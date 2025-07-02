# File: src/clustering/distance_metrics/binary.py

import numpy as np
from src.clustering.base_metric import DistanceMetric
from scipy.spatial.distance import hamming, jaccard

class Hamming(DistanceMetric):
    def __call__(self, x, y):
        return hamming(x, y)

class Jaccard(DistanceMetric):
    def __call__(self, x, y):
        return jaccard(x, y)

class Tanimoto(DistanceMetric):
    def __call__(self, x, y):
        dot = np.dot(x, y)
        return dot / (np.sum(x**2) + np.sum(y**2) - dot)

class Dice(DistanceMetric):
    def __call__(self, x, y):
        intersection = np.dot(x, y)
        return 1 - (2 * intersection) / (np.sum(x) + np.sum(y))

class SimpleMatching(DistanceMetric):
    def __call__(self, x, y):
        matches = np.sum(x == y)
        return 1 - matches / len(x)

class Levenshtein(DistanceMetric):
    def __call__(self, x, y):
        # Convert to strings for edit distance
        s1 = ''.join(str(int(i)) for i in x)
        s2 = ''.join(str(int(i)) for i in y)
        
        # Dynamic programming implementation
        if len(s1) < len(s2):
            return self(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

class DamerauLevenshtein(DistanceMetric):
    def __call__(self, x, y):
        # Convert to strings
        s1 = ''.join(str(int(i)) for i in x)
        s2 = ''.join(str(int(i)) for i in y)
        
        # Matrix initialization
        d = np.zeros((len(s1)+1, len(s2)+1))
        for i in range(len(s1)+1):
            d[i, 0] = i
        for j in range(len(s2)+1):
            d[0, j] = j
            
        # Populate matrix
        for i in range(1, len(s1)+1):
            for j in range(1, len(s2)+1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                d[i, j] = min(
                    d[i-1, j] + 1,    # deletion
                    d[i, j-1] + 1,    # insertion
                    d[i-1, j-1] + cost # substitution
                )
                # Transposition
                if (i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]):
                    d[i, j] = min(d[i, j], d[i-2, j-2] + cost)
        return d[len(s1), len(s2)]

class JaroWinkler(DistanceMetric):
    def __call__(self, x, y, winkler=True, scaling=0.1):
        # Convert to strings
        s1 = ''.join(str(int(i)) for i in x)
        s2 = ''.join(str(int(i)) for i in y)
        
        # Jaro distance
        match_distance = max(len(s1), len(s2)) // 2 - 1
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)
        
        matches = 0
        transpositions = 0
        
        for i in range(len(s1)):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len(s2))
            for j in range(start, end):
                if not s2_matches[j] and s1[i] == s2[j]:
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
                    
        if matches == 0:
            return 0.0
            
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
            
        jaro = ((matches / len(s1)) + 
                (matches / len(s2)) + 
                ((matches - transpositions/2) / matches)) / 3
        
        # Jaro-Winkler boost
        if winkler:
            prefix = 0
            for i in range(min(len(s1), len(s2), 4)):
                if s1[i] == s2[i]:
                    prefix += 1
                else:
                    break
            jaro = jaro + prefix * scaling * (1 - jaro)
            
        return 1 - jaro

class Ochiai(DistanceMetric):
    def __call__(self, x, y):
        intersection = np.dot(x, y)
        return 1 - intersection / np.sqrt(np.sum(x) * np.sum(y))