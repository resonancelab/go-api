package core

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// ResoLangAST represents the Abstract Syntax Tree for ResoLang programs
type ResoLangAST struct {
	Statements []Statement `json:"statements"`
}

// Statement represents a ResoLang statement
type Statement interface {
	Execute(ctx *ExecutionContext) error
	String() string
}

// ExecutionContext holds the execution context for ResoLang programs
type ExecutionContext struct {
	Variables          map[string]interface{}        `json:"variables"`
	PrimeOscillators   map[string]*PrimeOscillator   `json:"prime_oscillators"`
	QuaternionicStates map[string]*QuaternionicState `json:"quaternionic_states"`
	GlobalPhaseState   *GlobalPhaseState             `json:"global_phase_state"`
	Output             []string                      `json:"output"`
	Errors             []error                       `json:"errors"`
}

// ResoLangCompiler compiles ResoLang source code into executable AST
type ResoLangCompiler struct {
	source string
	tokens []Token
	pos    int
}

// Token represents a lexical token in ResoLang
type Token struct {
	Type   TokenType `json:"type"`
	Value  string    `json:"value"`
	Line   int       `json:"line"`
	Column int       `json:"column"`
}

// TokenType represents the type of a token
type TokenType int

const (
	TOKEN_EOF TokenType = iota
	TOKEN_IDENTIFIER
	TOKEN_NUMBER
	TOKEN_STRING
	TOKEN_OPERATOR
	TOKEN_KEYWORD
	TOKEN_PUNCTUATION
)

// Keywords and operators from Reson.net paper
const (
	KEYWORD_PRIMELET  = "primelet"
	KEYWORD_QUATSTATE = "quatstate"
	KEYWORD_EXECUTE   = "execute"
	KEYWORD_PAY       = "pay"
	KEYWORD_STORE     = "store"
	KEYWORD_RETRIEVE  = "retrieve"
	KEYWORD_RESONANT  = "resonant"
)

// PrimeletStatement represents a prime oscillator declaration
type PrimeletStatement struct {
	Name      string  `json:"name"`
	Prime     int     `json:"prime"`
	Amplitude float64 `json:"amplitude"`
	Phase     float64 `json:"phase"`
}

// QuatstateStatement represents a quaternionic state declaration
type QuatstateStatement struct {
	Name       string    `json:"name"`
	PrimeRef   string    `json:"prime_ref"`
	Gaussian   []float64 `json:"gaussian"`
	Eisenstein []float64 `json:"eisenstein"`
}

// ExecuteStatement represents program execution on nodes
type ExecuteStatement struct {
	Code  string   `json:"code"`
	Nodes []string `json:"nodes"`
}

// PayStatement represents RSN token payment
type PayStatement struct {
	Amount    float64 `json:"amount"`
	Currency  string  `json:"currency"`
	Recipient string  `json:"recipient"`
}

// StoreStatement represents holographic memory storage
type StoreStatement struct {
	Data   string `json:"data"`
	Memory string `json:"memory"`
}

// RetrieveStatement represents holographic memory retrieval
type RetrieveStatement struct {
	Data   string `json:"data"`
	Memory string `json:"memory"`
}

// ResonantBlock represents a resonant computation block
type ResonantBlock struct {
	Threshold  float64     `json:"threshold"`
	Statements []Statement `json:"statements"`
}

// NewResoLangCompiler creates a new ResoLang compiler
func NewResoLangCompiler() *ResoLangCompiler {
	return &ResoLangCompiler{}
}

// Compile compiles ResoLang source code into an AST
func (rlc *ResoLangCompiler) Compile(source string) (*ResoLangAST, error) {
	rlc.source = source
	rlc.tokens = rlc.tokenize(source)
	rlc.pos = 0

	ast := &ResoLangAST{
		Statements: make([]Statement, 0),
	}

	for rlc.currentToken().Type != TOKEN_EOF {
		stmt, err := rlc.parseStatement()
		if err != nil {
			return nil, err
		}
		ast.Statements = append(ast.Statements, stmt)
	}

	return ast, nil
}

// tokenize performs lexical analysis on the source code
func (rlc *ResoLangCompiler) tokenize(source string) []Token {
	tokens := make([]Token, 0)
	lines := strings.Split(source, "\n")

	lineNum := 1
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "%") {
			lineNum++
			continue
		}

		// Simple tokenization (can be enhanced with proper lexer)
		words := strings.Fields(line)
		colNum := 1

		for _, word := range words {
			token := Token{
				Line:   lineNum,
				Column: colNum,
			}

			// Identify token type
			if rlc.isKeyword(word) {
				token.Type = TOKEN_KEYWORD
				token.Value = word
			} else if rlc.isNumber(word) {
				token.Type = TOKEN_NUMBER
				token.Value = word
			} else if rlc.isIdentifier(word) {
				token.Type = TOKEN_IDENTIFIER
				token.Value = word
			} else if rlc.isOperator(word) {
				token.Type = TOKEN_OPERATOR
				token.Value = word
			} else {
				token.Type = TOKEN_STRING
				token.Value = word
			}

			tokens = append(tokens, token)
			colNum += len(word) + 1
		}

		lineNum++
	}

	// Add EOF token
	tokens = append(tokens, Token{Type: TOKEN_EOF, Value: ""})

	return tokens
}

// Helper methods for token classification
func (rlc *ResoLangCompiler) isKeyword(word string) bool {
	keywords := []string{
		KEYWORD_PRIMELET, KEYWORD_QUATSTATE, KEYWORD_EXECUTE,
		KEYWORD_PAY, KEYWORD_STORE, KEYWORD_RETRIEVE, KEYWORD_RESONANT,
	}

	for _, kw := range keywords {
		if word == kw {
			return true
		}
	}
	return false
}

func (rlc *ResoLangCompiler) isNumber(word string) bool {
	_, err := strconv.ParseFloat(word, 64)
	return err == nil
}

func (rlc *ResoLangCompiler) isIdentifier(word string) bool {
	match, _ := regexp.MatchString(`^[a-zA-Z_][a-zA-Z0-9_]*$`, word)
	return match
}

func (rlc *ResoLangCompiler) isOperator(word string) bool {
	operators := []string{"=", "+", "-", "*", "/", "(", ")", "[", "]", "{", "}", ","}
	for _, op := range operators {
		if word == op {
			return true
		}
	}
	return false
}

// Parser methods
func (rlc *ResoLangCompiler) currentToken() Token {
	if rlc.pos >= len(rlc.tokens) {
		return Token{Type: TOKEN_EOF}
	}
	return rlc.tokens[rlc.pos]
}

func (rlc *ResoLangCompiler) consumeToken() Token {
	token := rlc.currentToken()
	rlc.pos++
	return token
}

func (rlc *ResoLangCompiler) parseStatement() (Statement, error) {
	token := rlc.currentToken()

	switch token.Value {
	case KEYWORD_PRIMELET:
		return rlc.parsePrimeletStatement()
	case KEYWORD_QUATSTATE:
		return rlc.parseQuatstateStatement()
	case KEYWORD_EXECUTE:
		return rlc.parseExecuteStatement()
	case KEYWORD_PAY:
		return rlc.parsePayStatement()
	case KEYWORD_STORE:
		return rlc.parseStoreStatement()
	case KEYWORD_RETRIEVE:
		return rlc.parseRetrieveStatement()
	default:
		return nil, fmt.Errorf("unknown statement type: %s", token.Value)
	}
}

func (rlc *ResoLangCompiler) parsePrimeletStatement() (*PrimeletStatement, error) {
	rlc.consumeToken() // consume 'primelet'

	nameToken := rlc.consumeToken()
	if nameToken.Type != TOKEN_IDENTIFIER {
		return nil, fmt.Errorf("expected identifier after primelet")
	}

	if rlc.consumeToken().Value != "=" {
		return nil, fmt.Errorf("expected '=' after primelet name")
	}

	if rlc.consumeToken().Value != "oscillator" {
		return nil, fmt.Errorf("expected 'oscillator' after '='")
	}

	if rlc.consumeToken().Value != "(" {
		return nil, fmt.Errorf("expected '(' after oscillator")
	}

	// Parse parameters
	prime := 13      // default
	amplitude := 0.7 // default
	phase := 1.0     // default

	for rlc.currentToken().Value != ")" {
		param := rlc.consumeToken().Value
		if rlc.consumeToken().Value != "=" {
			return nil, fmt.Errorf("expected '=' after parameter name")
		}

		valueToken := rlc.consumeToken()
		switch param {
		case "prime":
			if p, err := strconv.Atoi(valueToken.Value); err == nil {
				prime = p
			}
		case "amplitude":
			if a, err := strconv.ParseFloat(valueToken.Value, 64); err == nil {
				amplitude = a
			}
		case "phase":
			if p, err := strconv.ParseFloat(valueToken.Value, 64); err == nil {
				phase = p
			}
		}

		if rlc.currentToken().Value == "," {
			rlc.consumeToken()
		}
	}

	rlc.consumeToken() // consume ')'

	return &PrimeletStatement{
		Name:      nameToken.Value,
		Prime:     prime,
		Amplitude: amplitude,
		Phase:     phase,
	}, nil
}

func (rlc *ResoLangCompiler) parseQuatstateStatement() (*QuatstateStatement, error) {
	rlc.consumeToken() // consume 'quatstate'

	nameToken := rlc.consumeToken()
	if nameToken.Type != TOKEN_IDENTIFIER {
		return nil, fmt.Errorf("expected identifier after quatstate")
	}

	if rlc.consumeToken().Value != "=" {
		return nil, fmt.Errorf("expected '=' after quatstate name")
	}

	if rlc.consumeToken().Value != "quaternion" {
		return nil, fmt.Errorf("expected 'quaternion' after '='")
	}

	if rlc.consumeToken().Value != "(" {
		return nil, fmt.Errorf("expected '(' after quaternion")
	}

	primeRef := rlc.consumeToken().Value

	if rlc.consumeToken().Value != "," {
		return nil, fmt.Errorf("expected ',' after prime reference")
	}

	// Parse gaussian and eisenstein coordinates
	gaussian := []float64{1.0, 2.0}   // defaults
	eisenstein := []float64{3.0, 4.0} // defaults

	for rlc.currentToken().Value != ")" {
		param := rlc.consumeToken().Value
		if rlc.consumeToken().Value != "=" {
			return nil, fmt.Errorf("expected '=' after parameter name")
		}

		if rlc.consumeToken().Value != "(" {
			return nil, fmt.Errorf("expected '(' after parameter value")
		}

		var coords []float64
		for rlc.currentToken().Value != ")" {
			if val, err := strconv.ParseFloat(rlc.consumeToken().Value, 64); err == nil {
				coords = append(coords, val)
			}
			if rlc.currentToken().Value == "," {
				rlc.consumeToken()
			}
		}
		rlc.consumeToken() // consume ')'

		switch param {
		case "gaussian":
			gaussian = coords
		case "eisenstein":
			eisenstein = coords
		}

		if rlc.currentToken().Value == "," {
			rlc.consumeToken()
		}
	}

	rlc.consumeToken() // consume ')'

	return &QuatstateStatement{
		Name:       nameToken.Value,
		PrimeRef:   primeRef,
		Gaussian:   gaussian,
		Eisenstein: eisenstein,
	}, nil
}

func (rlc *ResoLangCompiler) parseExecuteStatement() (*ExecuteStatement, error) {
	rlc.consumeToken() // consume 'execute'

	if rlc.consumeToken().Value != "{" {
		return nil, fmt.Errorf("expected '{' after execute")
	}

	// Parse code block (simplified)
	code := ""
	for rlc.currentToken().Value != "}" {
		code += rlc.consumeToken().Value + " "
	}
	rlc.consumeToken() // consume '}'

	if rlc.consumeToken().Value != "on" {
		return nil, fmt.Errorf("expected 'on' after execute block")
	}

	if rlc.consumeToken().Value != "nodes" {
		return nil, fmt.Errorf("expected 'nodes' after on")
	}

	if rlc.consumeToken().Value != "{" {
		return nil, fmt.Errorf("expected '{' after nodes")
	}

	var nodes []string
	for rlc.currentToken().Value != "}" {
		if rlc.currentToken().Type == TOKEN_IDENTIFIER {
			nodes = append(nodes, rlc.consumeToken().Value)
		} else {
			rlc.consumeToken()
		}
		if rlc.currentToken().Value == "," {
			rlc.consumeToken()
		}
	}
	rlc.consumeToken() // consume '}'

	return &ExecuteStatement{
		Code:  strings.TrimSpace(code),
		Nodes: nodes,
	}, nil
}

func (rlc *ResoLangCompiler) parsePayStatement() (*PayStatement, error) {
	rlc.consumeToken() // consume 'pay'

	amountToken := rlc.consumeToken()
	amount, err := strconv.ParseFloat(amountToken.Value, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid amount: %s", amountToken.Value)
	}

	currency := "RSN" // default
	currencyToken := rlc.consumeToken()
	if currencyToken.Value != "to" {
		currency = currencyToken.Value
		if rlc.consumeToken().Value != "to" {
			return nil, fmt.Errorf("expected 'to' after currency")
		}
	} else {
		// 'to' was consumed as currency, put it back
		rlc.pos--
	}

	recipient := rlc.consumeToken().Value

	return &PayStatement{
		Amount:    amount,
		Currency:  currency,
		Recipient: recipient,
	}, nil
}

func (rlc *ResoLangCompiler) parseStoreStatement() (*StoreStatement, error) {
	rlc.consumeToken() // consume 'store'

	data := rlc.consumeToken().Value

	if rlc.consumeToken().Value != "in" {
		return nil, fmt.Errorf("expected 'in' after data")
	}

	memory := rlc.consumeToken().Value

	return &StoreStatement{
		Data:   data,
		Memory: memory,
	}, nil
}

func (rlc *ResoLangCompiler) parseRetrieveStatement() (*RetrieveStatement, error) {
	rlc.consumeToken() // consume 'retrieve'

	data := rlc.consumeToken().Value

	if rlc.consumeToken().Value != "from" {
		return nil, fmt.Errorf("expected 'from' after data")
	}

	memory := rlc.consumeToken().Value

	return &RetrieveStatement{
		Data:   data,
		Memory: memory,
	}, nil
}

// Execution methods for statements
func (ps *PrimeletStatement) Execute(ctx *ExecutionContext) error {
	oscillator := NewPrimeOscillator(ps.Prime, ps.Amplitude, ps.Phase, 0.02)
	ctx.PrimeOscillators[ps.Name] = oscillator
	ctx.Output = append(ctx.Output, fmt.Sprintf("Created prime oscillator %s", ps.Name))
	return nil
}

func (ps *PrimeletStatement) String() string {
	return fmt.Sprintf("primelet %s = oscillator(prime=%d, amplitude=%.3f, phase=%.3f)",
		ps.Name, ps.Prime, ps.Amplitude, ps.Phase)
}

func (qs *QuatstateStatement) Execute(ctx *ExecutionContext) error {
	oscillator, exists := ctx.PrimeOscillators[qs.PrimeRef]
	if !exists {
		return fmt.Errorf("prime oscillator %s not found", qs.PrimeRef)
	}

	position := []float64{0.0, 0.0} // default position
	baseAmplitude := oscillator.GetComplexAmplitude()

	state := NewQuaternionicState(position, baseAmplitude, qs.Gaussian, qs.Eisenstein)
	state.PrimeOscillator = oscillator

	ctx.QuaternionicStates[qs.Name] = state
	ctx.Output = append(ctx.Output, fmt.Sprintf("Created quaternionic state %s", qs.Name))
	return nil
}

func (qs *QuatstateStatement) String() string {
	return fmt.Sprintf("quatstate %s = quaternion(%s, gaussian=%v, eisenstein=%v)",
		qs.Name, qs.PrimeRef, qs.Gaussian, qs.Eisenstein)
}

func (es *ExecuteStatement) Execute(ctx *ExecutionContext) error {
	ctx.Output = append(ctx.Output, fmt.Sprintf("Executing on nodes: %v", es.Nodes))
	ctx.Output = append(ctx.Output, fmt.Sprintf("Code: %s", es.Code))
	// In a full implementation, this would distribute execution across nodes
	return nil
}

func (es *ExecuteStatement) String() string {
	return fmt.Sprintf("execute { %s } on nodes %v", es.Code, es.Nodes)
}

func (ps *PayStatement) Execute(ctx *ExecutionContext) error {
	ctx.Output = append(ctx.Output, fmt.Sprintf("Paying %.2f %s to %s", ps.Amount, ps.Currency, ps.Recipient))
	// In a full implementation, this would interact with the token economy
	return nil
}

func (ps *PayStatement) String() string {
	return fmt.Sprintf("pay %.2f %s to %s", ps.Amount, ps.Currency, ps.Recipient)
}

func (ss *StoreStatement) Execute(ctx *ExecutionContext) error {
	ctx.Output = append(ctx.Output, fmt.Sprintf("Storing %s in %s", ss.Data, ss.Memory))
	// In a full implementation, this would interact with holographic memory
	return nil
}

func (ss *StoreStatement) String() string {
	return fmt.Sprintf("store %s in %s", ss.Data, ss.Memory)
}

func (rs *RetrieveStatement) Execute(ctx *ExecutionContext) error {
	ctx.Output = append(ctx.Output, fmt.Sprintf("Retrieving %s from %s", rs.Data, rs.Memory))
	// In a full implementation, this would interact with holographic memory
	return nil
}

func (rs *RetrieveStatement) String() string {
	return fmt.Sprintf("retrieve %s from %s", rs.Data, rs.Memory)
}

// ExecuteAST executes a complete ResoLang AST
func ExecuteAST(ast *ResoLangAST, globalPhaseState *GlobalPhaseState) (*ExecutionContext, error) {
	ctx := &ExecutionContext{
		Variables:          make(map[string]interface{}),
		PrimeOscillators:   make(map[string]*PrimeOscillator),
		QuaternionicStates: make(map[string]*QuaternionicState),
		GlobalPhaseState:   globalPhaseState,
		Output:             make([]string, 0),
		Errors:             make([]error, 0),
	}

	for _, stmt := range ast.Statements {
		if err := stmt.Execute(ctx); err != nil {
			ctx.Errors = append(ctx.Errors, err)
			return ctx, err
		}
	}

	return ctx, nil
}

// CompileAndExecute compiles and executes ResoLang source code
func CompileAndExecute(source string, globalPhaseState *GlobalPhaseState) (*ExecutionContext, error) {
	compiler := NewResoLangCompiler()
	ast, err := compiler.Compile(source)
	if err != nil {
		return nil, fmt.Errorf("compilation error: %w", err)
	}

	return ExecuteAST(ast, globalPhaseState)
}
