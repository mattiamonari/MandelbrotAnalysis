class MT19937:
    def __init__(self):
        # Period parameters
        self.N = 624
        self.M = 397
        self.MATRIX_A = 0x9908b0df   # constant vector a
        self.UPPER_MASK = 0x80000000  # most significant w-r bits
        self.LOWER_MASK = 0x7fffffff  # least significant r bits
        
        self.mt = [0] * self.N  # array for the state vector
        self.mti = self.N + 1   # mti==N+1 means mt[N] is not initialized
        
    def init_genrand(self, s: int) -> None:
        """Initialize with a seed"""
        self.mt[0] = s & 0xffffffff
        for self.mti in range(1, self.N):
            self.mt[self.mti] = (1812433253 * (self.mt[self.mti-1] ^ 
                                (self.mt[self.mti-1] >> 30)) + self.mti)
            self.mt[self.mti] &= 0xffffffff
    
    def init_by_array(self, init_key: list[int]) -> None:
        """Initialize by an array"""
        self.init_genrand(19650218)
        i, j = 1, 0
        k = max(self.N, len(init_key))
        
        for _ in range(k):
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i-1] ^ (self.mt[i-1] >> 30)) * 1664525))
                         + init_key[j] + j)
            self.mt[i] &= 0xffffffff
            i += 1
            j += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N-1]
                i = 1
            if j >= len(init_key):
                j = 0
                
        for _ in range(self.N-1):
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i-1] ^ (self.mt[i-1] >> 30)) * 1566083941))
                         - i)
            self.mt[i] &= 0xffffffff
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N-1]
                i = 1
        
        self.mt[0] = 0x80000000
        
    def genrand_int32(self) -> int:
        """Generate a random 32-bit integer"""
        mag01 = [0, self.MATRIX_A]
        
        if self.mti >= self.N:
            if self.mti == self.N + 1:
                self.init_genrand(5489)
            
            for kk in range(0, self.N-self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk+1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk+self.M] ^ (y >> 1) ^ mag01[y & 0x1]
            
            for kk in range(self.N-self.M, self.N-1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk+1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk+(self.M-self.N)] ^ (y >> 1) ^ mag01[y & 0x1]
                
            y = (self.mt[self.N-1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N-1] = self.mt[self.M-1] ^ (y >> 1) ^ mag01[y & 0x1]
            
            self.mti = 0
            
        y = self.mt[self.mti]
        self.mti += 1
        
        # Tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)
        
        return y & 0xffffffff
    
    def genrand_real2(self) -> float:
        """Generate a random number on [0,1)-real-interval"""
        return self.genrand_int32() * (1.0/4294967296.0)

class RandomSupport:
    def __init__(self, rng: MT19937):
        self.rng = rng
        self.MAX_UNSIGNED_RAND = 0xffffffff
        self.range = self.MAX_UNSIGNED_RAND
        self.reject_above = self.MAX_UNSIGNED_RAND
        self.divisor = 1
        
    def set_rand_range(self, r: int) -> int:
        """Set range for multiple draws"""
        N = self.MAX_UNSIGNED_RAND + 1
        if r:
            rp1 = r + 1
            self.divisor = N // rp1
            self.reject_above = rp1 * self.divisor - 1
            self.range = r
        return r
    
    def getrand_inrange(self, range_val: int) -> int:
        """Get a single random value in range"""
        if range_val == 0:
            return 0
            
        N = self.MAX_UNSIGNED_RAND + 1
        rp1 = range_val + 1
        divisor = N // rp1
        reject_above = rp1 * divisor - 1
        
        while True:
            r = self.rng.genrand_int32()
            if r <= reject_above:
                return r // divisor
    
    def permute(self, lst: list[int]) -> None:
        """Permute a list in place"""
        for i in range(len(lst)-1, 0, -1):
            r = self.getrand_inrange(i)
            lst[i], lst[r] = lst[r], lst[i]
