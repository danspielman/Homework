using Laplacians
using DataFrames
using DataStructures
using LightGraphs

# test the performance of CholLap and approxCholLap method.
function performanceTest(path; separator='\t')
  println("start task: test file ", path)
  # read in data and let indices start from 1
	iris = readtable(path, header = true, separator = separator, eltypes = [Int64, Int64])
	iris = convert(Matrix, iris) + 1
  # transfer these edges into sparse adjacency matrix, weight of each edge is 1
	m = maximum(iris)
	w = ones(length(iris[:, :1]))
	mat = sparse(vcat(iris[:, :1], iris[:, :2]), vcat(iris[:, :2], iris[:, :1]), vcat(w, w), m, m)
  # get the largest connected subgraph in original graph.
	matCopy = LightGraphs.DiGraph(mat)
	indices = strongly_connected_components(matCopy)
	sort!(indices, by = x -> length(x), rev = true)
	indices = sort!(indices[1])
	A = mat[indices, indices]
  # Use this matrix to test cholLap and approxCholLap.
	println("Begin sparsify method.")
  @time V = cholLap(A, tol=1e-2)
  println("End of CholLap experiment.");
	@time V = approxCholLap(A, tol=1e-2)
	println("End of approxCholLap experiment.")
end

performanceTest("ca-GrQc/data.txt")
