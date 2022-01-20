package index

type Candidate[U any] struct {
	Distance float32
	Item     U
}
