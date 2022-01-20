package index

type Candidate[U any] struct {
	Distance float64
	Item     U
}
