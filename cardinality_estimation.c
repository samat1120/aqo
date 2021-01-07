/*
 *******************************************************************************
 *
 *	CARDINALITY ESTIMATION
 *
 * This is the module in which cardinality estimation problem obtained from
 * cardinality_hooks turns into machine learning problem.
 *
 *******************************************************************************
 *
 * Copyright (c) 2016-2020, Postgres Professional
 *
 * IDENTIFICATION
 *	  aqo/cardinality_estimation.c
 *
 */

#include "aqo.h"
#include "optimizer/optimizer.h"

/*
 * General method for prediction the cardinality of given relation.
 */
double
predict_for_relation(List *restrict_clauses, List *selectivities,
					 List *relids, int *fss_hash)
{
	int	   nfeatures;
	int	   nrels;
	int	   ncols;
	int	   n_batches;
	double *matrix[n_all_samples];
	double targets[n_all_samples];
	double	*W1[WIDTH_1];
	double	*W1_m[WIDTH_1];
	double	*W1_v[WIDTH_1];
	double	*W2[WIDTH_2];
	double	*W2_m[WIDTH_2];
	double	*W2_v[WIDTH_2];
	double	W3[WIDTH_2];
	double	W3_m[WIDTH_2];
	double	W3_v[WIDTH_2];
	double	b1[WIDTH_1];
	double	b1_m[WIDTH_1];
	double	b1_v[WIDTH_1];
	double	b2[WIDTH_2];
	double	b2_m[WIDTH_2];
	double	b2_v[WIDTH_2];
	double	*feats, *fs;
	double	b3;
	double	b3_m;
	double	b3_v;
	int i, j, tmp;
	double	*features;
	int	*rels;
	int	*sorted_clauses;
	int	*hashes, *hshes;
	int *step_layer1;
	int steps;
	double	result;
	int to_add;
	to_add=0;
	for (i = 0; i < WIDTH_2; ++i){
	     W2[i] = palloc(sizeof(**W2) * WIDTH_1);
	     W2_m[i] = palloc(sizeof(**W2_m) * WIDTH_1);
	     W2_v[i] = palloc(sizeof(**W2_v) * WIDTH_1);}

	*fss_hash = get_fss_for_object(restrict_clauses, selectivities, relids, &nfeatures, &nrels, &features, &rels, &sorted_clauses);
	
	if ((nfeatures+nrels)>0){
	hshes = palloc0(sizeof(*hshes) * (nfeatures+nrels));
	fs = palloc0(sizeof(*fs) * (nfeatures+nrels));
	}
	for (i=0;i<nfeatures;i++){
		hshes[i] = sorted_clauses[i];
		fs[i] = features[i];
	}
	for (i=0;i<nrels;i++){
		hshes[i+nfeatures] = rels[i];
		fs[nfeatures+i] = 1;
	}

	if (load_fss(*fss_hash, &ncols, &n_batches, &hashes, matrix, targets, W1, W1_m, W1_v, W2, W2_m, W2_v, W3, W3_m, W3_v, b1, b1_m, b1_v, b2, b2_m, b2_v, &b3, &b3_m, &b3_v, &step_layer1, &steps)){
		if ((ncols+nfeatures+nrels)>0)
		   feats = palloc0(sizeof(*feats) * (ncols+nfeatures+nrels));
		for (i=0;i<(nfeatures+nrels);i++){
			tmp = i;
			for (j=0;j<ncols;j++){
				if(hashes[j]==hshes[i]){
					feats[j]=fs[i];
					++tmp;
					break;
				}
			}
			if (tmp==i){
				feats[ncols+to_add]=fs[i];
				++to_add;
			}
		}
		result = neural_predict (ncols, W1, b1, W2, b2, W3, b3, feats);
		for (j = ncols; j < (ncols+to_add); ++j){
			result = result + feats[j];
		}
		if (ncols > 0){
		    for (i = 0; i < WIDTH_1; ++i){
			pfree(W1[i]);
			pfree(W1_m[i]);
			pfree(W1_v[i]);
		    }
		    for (int i = 0; i < n_batches; ++i)
		        pfree(matrix[i]);
		    pfree(step_layer1);
		}
		if ((ncols+nfeatures+nrels)>0)
		    pfree(feats);
		if (ncols>0)
		    pfree(hashes);
	}
	else
	{
		/*
		 * Due to planning optimizer tries to build many alternate paths. Many
		 * of these not used in final query execution path. Consequently, only
		 * small part of paths was used for AQO learning and fetch into the AQO
		 * knowledge base.
		 */
		result = -1;
	}
	for (i = 0; i < WIDTH_2; ++i){
	     pfree(W2[i]);
	     pfree(W2_m[i]);
	     pfree(W2_v[i]);}

	if (nfeatures>0){
		pfree(features);
		pfree(sorted_clauses);}
	if (nrels>0)
		pfree(rels);
	if ((nfeatures+nrels)>0){
		pfree(hshes);
		pfree(fs);
	}

	if (result < 0)
		return -1;
	else
		return clamp_row_est(exp(result));
}
