package shared

import (
	"sync"
	"time"
)

// ResonanceCache provides caching functionality for the resonance system
type ResonanceCache struct {
	items     map[string]*CacheItem
	mutex     sync.RWMutex
	maxSize   int
	ttl       time.Duration
	cleanupInterval time.Duration
	stopCleanup     chan bool
}

// CacheItem represents an item in the cache
type CacheItem struct {
	Key        string
	Value      interface{}
	CreatedAt  time.Time
	AccessedAt time.Time
	AccessCount int64
	TTL        time.Duration
}

// NewResonanceCache creates a new resonance cache
func NewResonanceCache(maxSize int, ttl time.Duration) *ResonanceCache {
	cache := &ResonanceCache{
		items:           make(map[string]*CacheItem),
		maxSize:         maxSize,
		ttl:             ttl,
		cleanupInterval: ttl / 10, // Cleanup every 1/10th of TTL
		stopCleanup:     make(chan bool),
	}
	
	// Start cleanup goroutine
	go cache.cleanupLoop()
	
	return cache
}

// Set stores a value in the cache
func (rc *ResonanceCache) Set(key string, value interface{}) {
	rc.SetWithTTL(key, value, rc.ttl)
}

// SetWithTTL stores a value in the cache with custom TTL
func (rc *ResonanceCache) SetWithTTL(key string, value interface{}, ttl time.Duration) {
	rc.mutex.Lock()
	defer rc.mutex.Unlock()
	
	now := time.Now()
	
	// Check if we need to evict items to make room
	if len(rc.items) >= rc.maxSize {
		rc.evictLRU()
	}
	
	rc.items[key] = &CacheItem{
		Key:         key,
		Value:       value,
		CreatedAt:   now,
		AccessedAt:  now,
		AccessCount: 1,
		TTL:         ttl,
	}
}

// Get retrieves a value from the cache
func (rc *ResonanceCache) Get(key string) (interface{}, bool) {
	rc.mutex.Lock()
	defer rc.mutex.Unlock()
	
	item, exists := rc.items[key]
	if !exists {
		return nil, false
	}
	
	// Check if item has expired
	if time.Since(item.CreatedAt) > item.TTL {
		delete(rc.items, key)
		return nil, false
	}
	
	// Update access information
	item.AccessedAt = time.Now()
	item.AccessCount++
	
	return item.Value, true
}

// GetString retrieves a string value from the cache
func (rc *ResonanceCache) GetString(key string) (string, bool) {
	value, exists := rc.Get(key)
	if !exists {
		return "", false
	}
	
	if str, ok := value.(string); ok {
		return str, true
	}
	
	return "", false
}

// GetFloat64 retrieves a float64 value from the cache
func (rc *ResonanceCache) GetFloat64(key string) (float64, bool) {
	value, exists := rc.Get(key)
	if !exists {
		return 0.0, false
	}
	
	if f, ok := value.(float64); ok {
		return f, true
	}
	
	return 0.0, false
}

// GetInt retrieves an int value from the cache
func (rc *ResonanceCache) GetInt(key string) (int, bool) {
	value, exists := rc.Get(key)
	if !exists {
		return 0, false
	}
	
	if i, ok := value.(int); ok {
		return i, true
	}
	
	return 0, false
}

// Delete removes a value from the cache
func (rc *ResonanceCache) Delete(key string) {
	rc.mutex.Lock()
	defer rc.mutex.Unlock()
	
	delete(rc.items, key)
}

// Exists checks if a key exists in the cache
func (rc *ResonanceCache) Exists(key string) bool {
	rc.mutex.RLock()
	defer rc.mutex.RUnlock()
	
	item, exists := rc.items[key]
	if !exists {
		return false
	}
	
	// Check if item has expired
	if time.Since(item.CreatedAt) > item.TTL {
		return false
	}
	
	return true
}

// Clear removes all items from the cache
func (rc *ResonanceCache) Clear() {
	rc.mutex.Lock()
	defer rc.mutex.Unlock()
	
	rc.items = make(map[string]*CacheItem)
}

// Size returns the number of items in the cache
func (rc *ResonanceCache) Size() int {
	rc.mutex.RLock()
	defer rc.mutex.RUnlock()
	
	return len(rc.items)
}

// GetStats returns cache statistics
func (rc *ResonanceCache) GetStats() map[string]interface{} {
	rc.mutex.RLock()
	defer rc.mutex.RUnlock()
	
	totalAccesses := int64(0)
	oldestItem := time.Now()
	newestItem := time.Time{}
	
	for _, item := range rc.items {
		totalAccesses += item.AccessCount
		if item.CreatedAt.Before(oldestItem) {
			oldestItem = item.CreatedAt
		}
		if item.CreatedAt.After(newestItem) {
			newestItem = item.CreatedAt
		}
	}
	
	return map[string]interface{}{
		"total_items":     len(rc.items),
		"max_size":        rc.maxSize,
		"total_accesses":  totalAccesses,
		"oldest_item":     oldestItem,
		"newest_item":     newestItem,
		"ttl_seconds":     rc.ttl.Seconds(),
	}
}

// GetKeys returns all keys in the cache
func (rc *ResonanceCache) GetKeys() []string {
	rc.mutex.RLock()
	defer rc.mutex.RUnlock()
	
	keys := make([]string, 0, len(rc.items))
	for key := range rc.items {
		keys = append(keys, key)
	}
	
	return keys
}

// evictLRU evicts the least recently used item
func (rc *ResonanceCache) evictLRU() {
	if len(rc.items) == 0 {
		return
	}
	
	var lruKey string
	var lruTime time.Time = time.Now()
	
	for key, item := range rc.items {
		if item.AccessedAt.Before(lruTime) {
			lruTime = item.AccessedAt
			lruKey = key
		}
	}
	
	if lruKey != "" {
		delete(rc.items, lruKey)
	}
}

// cleanupLoop runs periodic cleanup of expired items
func (rc *ResonanceCache) cleanupLoop() {
	ticker := time.NewTicker(rc.cleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rc.cleanupExpired()
		case <-rc.stopCleanup:
			return
		}
	}
}

// cleanupExpired removes expired items from the cache
func (rc *ResonanceCache) cleanupExpired() {
	rc.mutex.Lock()
	defer rc.mutex.Unlock()
	
	now := time.Now()
	expiredKeys := make([]string, 0)
	
	for key, item := range rc.items {
		if now.Sub(item.CreatedAt) > item.TTL {
			expiredKeys = append(expiredKeys, key)
		}
	}
	
	for _, key := range expiredKeys {
		delete(rc.items, key)
	}
}

// Stop stops the cache cleanup goroutine
func (rc *ResonanceCache) Stop() {
	close(rc.stopCleanup)
}

// GetOrSet gets a value from cache or sets it if it doesn't exist
func (rc *ResonanceCache) GetOrSet(key string, valueFunc func() interface{}) interface{} {
	// Try to get first
	if value, exists := rc.Get(key); exists {
		return value
	}
	
	// Generate and set value
	value := valueFunc()
	rc.Set(key, value)
	return value
}

// GetOrSetWithTTL gets a value from cache or sets it with custom TTL
func (rc *ResonanceCache) GetOrSetWithTTL(key string, valueFunc func() interface{}, ttl time.Duration) interface{} {
	// Try to get first
	if value, exists := rc.Get(key); exists {
		return value
	}
	
	// Generate and set value
	value := valueFunc()
	rc.SetWithTTL(key, value, ttl)
	return value
}

// Increment increments a numeric value in the cache
func (rc *ResonanceCache) Increment(key string, delta float64) float64 {
	rc.mutex.Lock()
	defer rc.mutex.Unlock()
	
	item, exists := rc.items[key]
	if !exists {
		// Create new item with initial value
		now := time.Now()
		rc.items[key] = &CacheItem{
			Key:         key,
			Value:       delta,
			CreatedAt:   now,
			AccessedAt:  now,
			AccessCount: 1,
			TTL:         rc.ttl,
		}
		return delta
	}
	
	// Update existing item
	if currentValue, ok := item.Value.(float64); ok {
		newValue := currentValue + delta
		item.Value = newValue
		item.AccessedAt = time.Now()
		item.AccessCount++
		return newValue
	}
	
	// If existing value is not float64, replace it
	item.Value = delta
	item.AccessedAt = time.Now()
	item.AccessCount++
	return delta
}

// Decrement decrements a numeric value in the cache
func (rc *ResonanceCache) Decrement(key string, delta float64) float64 {
	return rc.Increment(key, -delta)
}