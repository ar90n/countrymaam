package index

type Candidate[U comparable] struct {
	Distance float64
	Item     U
}
