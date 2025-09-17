package validation

import (
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strings"
	"time"
)

// ValidationError represents a validation error
type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Value   interface{} `json:"value,omitempty"`
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("validation failed for field '%s': %s", e.Field, e.Message)
}

// ValidationResult holds validation results
type ValidationResult struct {
	Valid  bool              `json:"valid"`
	Errors []ValidationError `json:"errors,omitempty"`
}

// Validator provides validation functions
type Validator struct {
	errors []ValidationError
}

// NewValidator creates a new validator instance
func NewValidator() *Validator {
	return &Validator{
		errors: make([]ValidationError, 0),
	}
}

// ValidateRequired checks if a value is present
func (v *Validator) ValidateRequired(field string, value interface{}) *Validator {
	if value == nil {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "field is required",
			Value:   value,
		})
		return v
	}

	switch val := value.(type) {
	case string:
		if strings.TrimSpace(val) == "" {
			v.errors = append(v.errors, ValidationError{
				Field:   field,
				Message: "field cannot be empty",
				Value:   value,
			})
		}
	case []interface{}:
		if len(val) == 0 {
			v.errors = append(v.errors, ValidationError{
				Field:   field,
				Message: "array cannot be empty",
				Value:   value,
			})
		}
	case map[string]interface{}:
		if len(val) == 0 {
			v.errors = append(v.errors, ValidationError{
				Field:   field,
				Message: "object cannot be empty",
				Value:   value,
			})
		}
	}

	return v
}

// ValidateRange checks if a numeric value is within range
func (v *Validator) ValidateRange(field string, value interface{}, min, max float64) *Validator {
	var numValue float64
	var valid bool

	switch val := value.(type) {
	case int:
		numValue = float64(val)
		valid = true
	case int32:
		numValue = float64(val)
		valid = true
	case int64:
		numValue = float64(val)
		valid = true
	case float32:
		numValue = float64(val)
		valid = true
	case float64:
		numValue = val
		valid = true
	}

	if !valid {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "field must be a number",
			Value:   value,
		})
		return v
	}

	if numValue < min || numValue > max {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: fmt.Sprintf("value must be between %g and %g", min, max),
			Value:   value,
		})
	}

	return v
}

// ValidatePositive checks if a numeric value is positive
func (v *Validator) ValidatePositive(field string, value interface{}) *Validator {
	return v.ValidateRange(field, value, 0.000001, math.Inf(1))
}

// ValidateArrayLength checks array length constraints
func (v *Validator) ValidateArrayLength(field string, value interface{}, minLen, maxLen int) *Validator {
	var length int
	var valid bool

	switch val := value.(type) {
	case []interface{}:
		length = len(val)
		valid = true
	case []string:
		length = len(val)
		valid = true
	case []int:
		length = len(val)
		valid = true
	case []float64:
		length = len(val)
		valid = true
	default:
		// Use reflection for other slice types
		rv := reflect.ValueOf(value)
		if rv.Kind() == reflect.Slice {
			length = rv.Len()
			valid = true
		}
	}

	if !valid {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "field must be an array",
			Value:   value,
		})
		return v
	}

	if length < minLen {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: fmt.Sprintf("array must have at least %d elements", minLen),
			Value:   value,
		})
	}

	if maxLen > 0 && length > maxLen {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: fmt.Sprintf("array cannot have more than %d elements", maxLen),
			Value:   value,
		})
	}

	return v
}

// ValidateStringLength checks string length constraints
func (v *Validator) ValidateStringLength(field string, value interface{}, minLen, maxLen int) *Validator {
	str, ok := value.(string)
	if !ok {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "field must be a string",
			Value:   value,
		})
		return v
	}

	length := len(str)
	if length < minLen {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: fmt.Sprintf("string must be at least %d characters", minLen),
			Value:   value,
		})
	}

	if maxLen > 0 && length > maxLen {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: fmt.Sprintf("string cannot be longer than %d characters", maxLen),
			Value:   value,
		})
	}

	return v
}

// ValidatePattern checks if a string matches a regex pattern
func (v *Validator) ValidatePattern(field string, value interface{}, pattern string, description string) *Validator {
	str, ok := value.(string)
	if !ok {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "field must be a string",
			Value:   value,
		})
		return v
	}

	matched, err := regexp.MatchString(pattern, str)
	if err != nil {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "invalid pattern for validation",
			Value:   value,
		})
		return v
	}

	if !matched {
		message := fmt.Sprintf("field does not match required pattern")
		if description != "" {
			message = fmt.Sprintf("field must be %s", description)
		}
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: message,
			Value:   value,
		})
	}

	return v
}

// ValidateEmail checks if a string is a valid email
func (v *Validator) ValidateEmail(field string, value interface{}) *Validator {
	emailPattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
	return v.ValidatePattern(field, value, emailPattern, "a valid email address")
}

// ValidateEnum checks if a value is in allowed enum values
func (v *Validator) ValidateEnum(field string, value interface{}, allowedValues []string) *Validator {
	str, ok := value.(string)
	if !ok {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "field must be a string",
			Value:   value,
		})
		return v
	}

	for _, allowed := range allowedValues {
		if str == allowed {
			return v
		}
	}

	v.errors = append(v.errors, ValidationError{
		Field:   field,
		Message: fmt.Sprintf("value must be one of: %s", strings.Join(allowedValues, ", ")),
		Value:   value,
	})

	return v
}

// ValidateTimeout checks if a timeout value is reasonable
func (v *Validator) ValidateTimeout(field string, value interface{}) *Validator {
	var seconds float64
	var valid bool

	switch val := value.(type) {
	case int:
		seconds = float64(val)
		valid = true
	case float64:
		seconds = val
		valid = true
	case time.Duration:
		seconds = val.Seconds()
		valid = true
	}

	if !valid {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "timeout must be a number or duration",
			Value:   value,
		})
		return v
	}

	// Validate reasonable timeout ranges (1 second to 1 hour)
	if seconds < 1 {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "timeout must be at least 1 second",
			Value:   value,
		})
	}

	if seconds > 3600 {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: "timeout cannot exceed 1 hour",
			Value:   value,
		})
	}

	return v
}

// ValidatePrimes checks if an array contains valid prime numbers
func (v *Validator) ValidatePrimes(field string, value interface{}) *Validator {
	primes, ok := value.([]int)
	if !ok {
		// Try to convert from []interface{}
		if arr, ok := value.([]interface{}); ok {
			primes = make([]int, len(arr))
			for i, val := range arr {
				if intVal, ok := val.(int); ok {
					primes[i] = intVal
				} else if floatVal, ok := val.(float64); ok {
					primes[i] = int(floatVal)
				} else {
					v.errors = append(v.errors, ValidationError{
						Field:   field,
						Message: "all array elements must be integers",
						Value:   value,
					})
					return v
				}
			}
		} else {
			v.errors = append(v.errors, ValidationError{
				Field:   field,
				Message: "field must be an array of integers",
				Value:   value,
			})
			return v
		}
	}

	// Check each number is prime
	for i, num := range primes {
		if !isPrime(num) {
			v.errors = append(v.errors, ValidationError{
				Field:   field,
				Message: fmt.Sprintf("element at index %d (%d) is not a prime number", i, num),
				Value:   value,
			})
		}
	}

	return v
}

// ValidateQuantumState checks if quantum state parameters are valid
func (v *Validator) ValidateQuantumState(field string, amplitudes interface{}) *Validator {
	arr, ok := amplitudes.([]complex128)
	if !ok {
		// Try to convert from other types
		if floatArr, ok := amplitudes.([]float64); ok {
			arr = make([]complex128, len(floatArr))
			for i, val := range floatArr {
				arr[i] = complex(val, 0)
			}
		} else {
			v.errors = append(v.errors, ValidationError{
				Field:   field,
				Message: "quantum state amplitudes must be complex numbers",
				Value:   amplitudes,
			})
			return v
		}
	}

	// Check normalization
	sumSquares := 0.0
	for _, amp := range arr {
		magnitude := real(amp)*real(amp) + imag(amp)*imag(amp)
		sumSquares += magnitude
	}

	if math.Abs(sumSquares-1.0) > 1e-6 {
		v.errors = append(v.errors, ValidationError{
			Field:   field,
			Message: fmt.Sprintf("quantum state must be normalized (sum of squares = %f)", sumSquares),
			Value:   amplitudes,
		})
	}

	return v
}

// Result returns the validation result
func (v *Validator) Result() ValidationResult {
	return ValidationResult{
		Valid:  len(v.errors) == 0,
		Errors: v.errors,
	}
}

// HasErrors returns true if there are validation errors
func (v *Validator) HasErrors() bool {
	return len(v.errors) > 0
}

// GetErrors returns all validation errors
func (v *Validator) GetErrors() []ValidationError {
	return v.errors
}

// isPrime checks if a number is prime
func isPrime(n int) bool {
	if n < 2 {
		return false
	}
	if n == 2 {
		return true
	}
	if n%2 == 0 {
		return false
	}
	for i := 3; i*i <= n; i += 2 {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// Common validation functions for quantum platform

// ValidateResonanceConfig validates resonance configuration parameters
func ValidateResonanceConfig(config map[string]interface{}) ValidationResult {
	v := NewValidator()

	if maxIterations, exists := config["max_iterations"]; exists {
		v.ValidateRange("max_iterations", maxIterations, 1, 100000)
	}

	if timeout, exists := config["timeout_seconds"]; exists {
		v.ValidateTimeout("timeout_seconds", timeout)
	}

	if threshold, exists := config["threshold"]; exists {
		v.ValidateRange("threshold", threshold, 0, 1)
	}

	if precision, exists := config["precision_level"]; exists {
		v.ValidateEnum("precision_level", precision, []string{"low", "medium", "high", "ultra"})
	}

	return v.Result()
}

// ValidateQuantumParameters validates quantum simulation parameters
func ValidateQuantumParameters(params map[string]interface{}) ValidationResult {
	v := NewValidator()

	if primes, exists := params["primes"]; exists {
		v.ValidateRequired("primes", primes)
		v.ValidatePrimes("primes", primes)
		v.ValidateArrayLength("primes", primes, 1, 100)
	}

	if steps, exists := params["steps"]; exists {
		v.ValidateRange("steps", steps, 1, 100000)
	}

	if lambda, exists := params["lambda"]; exists {
		v.ValidateRange("lambda", lambda, 0.001, 1.0)
	}

	return v.Result()
}