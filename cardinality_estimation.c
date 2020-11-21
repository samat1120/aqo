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
	double	**W1;
	double	**new_W1;
	double	**W2;
	double	*W3;
	double	*b1;
	double	*b2;
	double	*feats, *fs;
	double	b3 = 0;
	double	*features;
	double stdv;
	int	*rels;
	int	*sorted_clauses;
	int	*hashes, *hshes;
	double	result;
	int		i,j,tmp;
	int to_add=0;

	*fss_hash = get_fss_for_object(restrict_clauses, selectivities, relids,
														&nfeatures, &nrels, &features, &rels, &sorted_clauses);

	W1 = (double**)palloc0(sizeof(double*) * WIDTH_1);
	W2 = (double**)palloc0(sizeof(double*) * WIDTH_2);
	for (i = 0; i < WIDTH_2; ++i)
			W2[i] = palloc0(sizeof(**W2) * WIDTH_1);
	W3 = palloc0(sizeof(*W3) * WIDTH_2);
	b1 = palloc0(sizeof(*b1) * WIDTH_1);
	b2 = palloc0(sizeof(*b2) * WIDTH_2);
	hshes = palloc0(sizeof(*hshes) * (nfeatures+nrels));
	fs = palloc0(sizeof(*fs) * (nfeatures+nrels));
	for (i=0;i<nfeatures;i++){
		hshes[i] = sorted_clauses[i];
		fs[i] = features[i];
	}
	for (i=0;i<nrels;i++){
		hshes[i+nfeatures] = rels[i];
		fs[nfeatures+i] = 1;
	}

	if (load_fss(*fss_hash, &ncols, &hashes, W1, W2, W3, b1, b2, b3)){
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
		feats = repalloc(feats, (ncols+to_add) * sizeof(*feats));
		new_W1 = (double**)palloc0(sizeof(double*) * WIDTH_1);
		srand(1);
		stdv = 1 / sqrt(ncols+to_add);
		for (i = 0; i < WIDTH_1; ++i)
			new_W1[i] = palloc0(sizeof(**new_W1) * (ncols+to_add));
		for (j = 0; j < (ncols+to_add); ++j){
			for (i = 0; i < WIDTH_1; ++i){
				new_W1[i][j] = (stdv + stdv)*(rand()/(double)RAND_MAX) - stdv;
			}
		}
		for (i = 0; i < WIDTH_1; ++i){
			for (j = 0; j < ncols; ++j){
				new_W1[i][j] = W1[i][j];
			}
		}
		result = neural_predict ((ncols+to_add), new_W1, b1, W2, b2, W3, b3, feats);
		if (ncols > 0)
				for (i = 0; i < WIDTH_1; ++i){
					pfree(W1[i]);
				}
		if ((ncols+to_add) > 0)
				for (i = 0; i < WIDTH_1; ++i){
					pfree(new_W1[i]);
				}
		pfree(feats);
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

	pfree(features);
	pfree(rels);
	pfree(sorted_clauses);
	for (i = 0; i < WIDTH_2; ++i)
		pfree(W2[i]);
	pfree(W3);
	pfree(b1);
	pfree(b2);
	pfree(hshes);
	pfree(fs);

	if (result < 0)
		return -1;
	else
		return clamp_row_est(exp(result));
}
