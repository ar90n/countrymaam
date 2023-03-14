package countrymaam

import "errors"

var (
	ErrInvalidFeaturesAndItems = errors.New("invalid features and items")
	ErrInvalidFeatureDim       = errors.New("invalid feature dim")
)
