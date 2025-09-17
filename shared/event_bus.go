package shared

import (
	"sync"
	"time"
)

// EventBus handles event publishing and subscription
type EventBus struct {
	subscribers map[string][]EventHandler
	mutex       sync.RWMutex
	eventQueue  chan Event
	workers     int
	stopWorkers chan bool
	running     bool
	runMutex    sync.Mutex
}

// Event represents an event in the system
type Event struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
	ID        string                 `json:"id"`
}

// EventHandler is a function that handles events
type EventHandler func(event Event)

// NewEventBus creates a new event bus
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]EventHandler),
		eventQueue:  make(chan Event, 1000), // Buffer for 1000 events
		workers:     5,                      // Number of worker goroutines
		stopWorkers: make(chan bool),
		running:     false,
	}
}

// Start starts the event bus workers
func (eb *EventBus) Start() {
	eb.runMutex.Lock()
	defer eb.runMutex.Unlock()
	
	if eb.running {
		return
	}
	
	eb.running = true
	
	// Start worker goroutines
	for i := 0; i < eb.workers; i++ {
		go eb.worker()
	}
}

// Stop stops the event bus workers
func (eb *EventBus) Stop() {
	eb.runMutex.Lock()
	defer eb.runMutex.Unlock()
	
	if !eb.running {
		return
	}
	
	eb.running = false
	
	// Stop all workers
	for i := 0; i < eb.workers; i++ {
		eb.stopWorkers <- true
	}
	
	close(eb.eventQueue)
}

// Subscribe subscribes to events of a specific type
func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	if eb.subscribers[eventType] == nil {
		eb.subscribers[eventType] = make([]EventHandler, 0)
	}
	
	eb.subscribers[eventType] = append(eb.subscribers[eventType], handler)
}

// Unsubscribe removes all handlers for an event type
func (eb *EventBus) Unsubscribe(eventType string) {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	delete(eb.subscribers, eventType)
}

// Publish publishes an event
func (eb *EventBus) Publish(event Event) {
	// Set timestamp if not set
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	
	// Generate ID if not set
	if event.ID == "" {
		event.ID = eb.generateEventID()
	}
	
	// Add to queue
	select {
	case eb.eventQueue <- event:
		// Event queued successfully
	default:
		// Queue is full, drop event (or handle overflow)
	}
}

// PublishSync publishes an event synchronously
func (eb *EventBus) PublishSync(event Event) {
	// Set timestamp if not set
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	
	// Generate ID if not set
	if event.ID == "" {
		event.ID = eb.generateEventID()
	}
	
	// Handle immediately
	eb.handleEvent(event)
}

// worker processes events from the queue
func (eb *EventBus) worker() {
	for {
		select {
		case event := <-eb.eventQueue:
			eb.handleEvent(event)
		case <-eb.stopWorkers:
			return
		}
	}
}

// handleEvent handles a single event
func (eb *EventBus) handleEvent(event Event) {
	eb.mutex.RLock()
	handlers := eb.subscribers[event.Type]
	
	// Also check for wildcard subscribers
	wildcardHandlers := eb.subscribers["*"]
	
	// Combine handlers
	allHandlers := make([]EventHandler, 0, len(handlers)+len(wildcardHandlers))
	allHandlers = append(allHandlers, handlers...)
	allHandlers = append(allHandlers, wildcardHandlers...)
	
	eb.mutex.RUnlock()
	
	// Execute handlers
	for _, handler := range allHandlers {
		// Execute in goroutine to prevent blocking
		go func(h EventHandler) {
			defer func() {
				if r := recover(); r != nil {
					// Log panic and continue
				}
			}()
			h(event)
		}(handler)
	}
}

// generateEventID generates a unique event ID
func (eb *EventBus) generateEventID() string {
	return time.Now().Format("20060102150405") + "-" + randString(8)
}

// randString generates a random string of specified length
func randString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	result := make([]byte, length)
	for i := range result {
		result[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(result)
}

// GetSubscriberCount returns the number of subscribers for an event type
func (eb *EventBus) GetSubscriberCount(eventType string) int {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	return len(eb.subscribers[eventType])
}

// GetTotalSubscribers returns the total number of subscribers
func (eb *EventBus) GetTotalSubscribers() int {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	total := 0
	for _, handlers := range eb.subscribers {
		total += len(handlers)
	}
	
	return total
}

// GetEventTypes returns all subscribed event types
func (eb *EventBus) GetEventTypes() []string {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	types := make([]string, 0, len(eb.subscribers))
	for eventType := range eb.subscribers {
		types = append(types, eventType)
	}
	
	return types
}

// GetQueueSize returns the current size of the event queue
func (eb *EventBus) GetQueueSize() int {
	return len(eb.eventQueue)
}

// GetStats returns event bus statistics
func (eb *EventBus) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"total_subscribers": eb.GetTotalSubscribers(),
		"event_types":       len(eb.subscribers),
		"queue_size":        eb.GetQueueSize(),
		"queue_capacity":    cap(eb.eventQueue),
		"workers":           eb.workers,
		"running":           eb.running,
	}
}