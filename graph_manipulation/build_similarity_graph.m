function [W] = build_similarity_graph(user_id, film_id, ratings, graph_param)
%  [W] = build_similarity_graph(X, graph_param, graph_similarity_function)
%      Computes the similarity matrix for a given dataset of samples.
%
%  Input
%  user_id : user ID
%  film_id : films ID
%  ratings : ratings of users
%  graph_param:
%      structure containing the graph construction parameters as fields
%  graph_param.graph_type:
%      knn or eps graph, as a string, controls the graph that
%      the function will produce
%  graph_param.graph_thresh:
%      controls the main parameter of the graph, the number
%      of neighbours k for k-nn, and the threshold eps for epsilon graphs
%  graph_param.sigma2:
%      the sigma value for the exponential function, already squared
%  graph_similarity_function:
%      the similarity function between points, defaults to inverse exponential.
%
%  Output
%  W:
%      (n x n) dimensional matrix representing the adjacency matrix of the graph


% unpack the type of the graph to build and the respective      %
% threshold and similarity function options     
unique_users = unique(user_id);
num_users = size(unique_users,1);
unique_films = unique(user_id);
num_films = size(unique_films,1);
X = zeros(num_users,num_films);
for i = 1:num_users
    X(i,film_id(find(user_id == unique_users(i)))) = ratings(find(user_id == unique_users(i)));
end

intermediaite_similarity_graph = pdist2(X, X,'cosine');
graph_type = graph_param.graph_type;
graph_thresh = graph_param.graph_thresh; % the number of neighbours for the graph
sigma2 = graph_param.sigma2; % exponential_euclidean's sigma^2

W = zeros(num_users,num_users);
if strcmp(graph_type,'knn') == 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  compute a k-nn graph from the similarities                   %
    %  for each node x_i, a k-nn graph has weights                  %
    %  w_ij = d(x_i,x_j) for the k closest nodes to x_i, and 0      %
    %  for all the k-n remaining nodes                              %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [D,I] = sort(intermediaite_similarity_graph,2,'descend');

    i_index = I(:, 1:graph_thresh);
    j_index = repmat([1:num_users]', 1,graph_thresh);
    z_values = D(:, 1:graph_thresh);

    W(sub2ind(size(W),j_index(:),i_index(:))) = z_values(:);
    W(sub2ind(size(W),i_index(:),j_index(:))) = z_values(:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif strcmp(graph_type,'eps') == 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  compute an epsilon graph from the similarities               %
    %  for each node x_i, an epsilon graph has weights              %
    %  w_ij = d(x_i,x_j) when w_ij > eps, and 0 otherwise           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    W(intermediaite_similarity_graph >= graph_thresh) = intermediaite_similarity_graph(intermediaite_similarity_graph >= graph_thresh);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else

    error('build_similarity_graph: not a valid graph type')

end
