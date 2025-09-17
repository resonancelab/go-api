package iching

// Complete the GetWisdomInsights method
func (ic *IChingEngine) GetWisdomInsights() []*Insight {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	if ic.wisdomMatrix != nil {
		insights := make([]*Insight, len(ic.wisdomMatrix.Insights))
		copy(insights, ic.wisdomMatrix.Insights)
		return insights
	}
	
	return make([]*Insight, 0)
}

// generateCommentary creates contextual commentary for a hexagram
func (ic *IChingEngine) generateCommentary(hexagramState *HexagramState, question string) string {
	// Generate commentary based on question and hexagram meaning
	if question == "" {
		return hexagramState.Meaning
	}
	
	commentary := fmt.Sprintf("In relation to your question about %s, %s suggests %s. %s", 
		ic.extractQuestionTopic(question),
		hexagramState.Name,
		hexagramState.Meaning,
		hexagramState.Judgment)
	
	return commentary
}

// extractQuestionTopic extracts the main topic from a question
func (ic *IChingEngine) extractQuestionTopic(question string) string {
	words := strings.Fields(strings.ToLower(question))
	
	// Look for key topic words
	topics := map[string]string{
		"love":         "love and relationships",
		"relationship": "love and relationships", 
		"career":       "career and professional life",
		"job":          "career and professional life",
		"work":         "work and professional matters",
		"health":       "health and well-being",
		"money":        "financial matters",
		"finance":      "financial matters",
		"family":       "family relationships",
		"future":       "future possibilities",
		"decision":     "decision making",
		"change":       "life changes",
		"travel":       "travel and movement",
		"study":        "learning and education",
		"business":     "business ventures",
		"marriage":     "marriage and commitment",
		"friendship":   "friendships and social connections",
	}
	
	for _, word := range words {
		if topic, exists := topics[word]; exists {
			return topic
		}
	}
	
	return "your current situation"
}

// generateAdvice creates specific advice for the context
func (ic *IChingEngine) generateAdvice(hexagramState *HexagramState, context string) string {
	contextAdvice := map[string][]string{
		"love": {
			"Open your heart to genuine connection",
			"Balance independence with partnership", 
			"Communicate with honesty and compassion",
			"Trust in the timing of love",
		},
		"career": {
			"Align your work with your true purpose",
			"Build skills through continuous learning",
			"Collaborate while maintaining integrity",
			"Seek opportunities for growth",
		},
		"health": {
			"Listen to your body's wisdom",
			"Maintain balance in all things",
			"Nurture both physical and mental well-being",
			"Embrace natural healing approaches",
		},
		"general": {
			"Follow the path of wisdom and virtue",
			"Remain flexible while staying true to principles",
			"Seek harmony in all relationships",
			"Trust in the natural order of things",
		},
	}
	
	advice, exists := contextAdvice[context]
	if !exists {
		advice = contextAdvice["general"]
	}
	
	// Select advice based on hexagram characteristics
	index := hexagramState.HexagramNumber % len(advice)
	return advice[index]
}

// generateWarning creates warnings based on hexagram
func (ic *IChingEngine) generateWarning(hexagramState *HexagramState) string {
	warnings := []string{
		"Beware of acting without proper consideration",
		"Avoid extremes and seek the middle path",
		"Don't force outcomes against natural timing",
		"Guard against pride and overconfidence", 
		"Be cautious of deception or illusion",
		"Avoid hasty decisions in important matters",
		"Don't neglect your responsibilities",
		"Be mindful of the consequences of your actions",
	}
	
	// Select warning based on hexagram number
	index := (hexagramState.HexagramNumber + 1) % len(warnings)
	return warnings[index]
}

// generateOpportunity creates opportunity guidance
func (ic *IChingEngine) generateOpportunity(hexagramState *HexagramState) string {
	opportunities := []string{
		"A new beginning offers fresh possibilities",
		"Unexpected doors may open for growth", 
		"Your talents can find new expression",
		"Relationships may deepen and strengthen",
		"Creative solutions will emerge from challenges",
		"Leadership opportunities await your initiative",
		"Learning experiences will expand your wisdom",
		"Spiritual insights may guide your path",
	}
	
	// Select opportunity based on hexagram characteristics
	index := (hexagramState.HexagramNumber + 2) % len(opportunities)
	return opportunities[index]
}

// generateChallenge creates challenge guidance
func (ic *IChingEngine) generateChallenge(hexagramState *HexagramState) string {
	challenges := []string{
		"Patience will be required for lasting success",
		"Old patterns must be released for new growth",
		"Difficult conversations may be necessary",
		"Inner work is needed before outer change",
		"Perseverance through obstacles builds character",
		"Balancing competing demands requires wisdom",
		"Letting go of control allows natural flow",
		"Facing fears opens the door to transformation",
	}
	
	// Select challenge based on hexagram properties
	index := (hexagramState.HexagramNumber + 4) % len(challenges)
	return challenges[index]
}