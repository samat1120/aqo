-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION aqo" to load this file. \quit

CREATE TABLE public.aqo_queries (
	query_hash		int PRIMARY KEY,
	learn_aqo		boolean NOT NULL,
	use_aqo			boolean NOT NULL,
	fspace_hash		int NOT NULL,
	auto_tuning		boolean NOT NULL
);

CREATE TABLE public.aqo_query_texts (
	query_hash		int PRIMARY KEY REFERENCES public.aqo_queries ON DELETE CASCADE,
	query_text		varchar NOT NULL
);

CREATE TABLE public.aqo_query_stat (
	query_hash		int PRIMARY KEY REFERENCES public.aqo_queries ON DELETE CASCADE,
	execution_time_with_aqo					double precision[],
	execution_time_without_aqo				double precision[],
	planning_time_with_aqo					double precision[],
	planning_time_without_aqo				double precision[],
	cardinality_error_with_aqo				double precision[],
	cardinality_error_without_aqo			double precision[],
	executions_with_aqo						bigint,
	executions_without_aqo					bigint
);

CREATE TABLE public.aqo_data (
	fspace_hash		int NOT NULL REFERENCES public.aqo_queries ON DELETE CASCADE,
	fsspace_hash	int NOT NULL,
	nfeatures		int NOT NULL,
	n_batch                 int NOT NULL,
	matrix                  double precision[][],
	targets                 double precision[],
	W1                      double precision[][],
	W1_m                    double precision[][],
	W1_v                    double precision[][],
	W2                      double precision[][],
	W2_m                    double precision[][],
	W2_v                    double precision[][],
	W3                      double precision[],
	W3_m                    double precision[],
	W3_v                    double precision[],
	b1                      double precision[],
	b1_m                    double precision[],
	b1_v                    double precision[],
	b2                      double precision[],
	b2_m                    double precision[],
	b2_v                    double precision[],
	b3                      double precision,
	b3_m                    double precision,
	b3_v                    double precision,
	hashes 			integer[],
	steps_layer1            integer[],
	step                    int NOT NULL
	UNIQUE (fspace_hash, fsspace_hash)
);

CREATE INDEX aqo_queries_query_hash_idx ON public.aqo_queries (query_hash);
CREATE INDEX aqo_query_texts_query_hash_idx ON public.aqo_query_texts (query_hash);
CREATE INDEX aqo_query_stat_idx ON public.aqo_query_stat (query_hash);
CREATE INDEX aqo_fss_access_idx ON public.aqo_data (fspace_hash, fsspace_hash);

INSERT INTO public.aqo_queries VALUES (0, false, false, 0, false);
INSERT INTO public.aqo_query_texts VALUES (0, 'COMMON feature space (do not delete!)');
-- a virtual query for COMMON feature space

CREATE FUNCTION invalidate_deactivated_queries_cache() RETURNS trigger
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE TRIGGER aqo_queries_invalidate AFTER UPDATE OR DELETE OR TRUNCATE
	ON public.aqo_queries FOR EACH STATEMENT
	EXECUTE PROCEDURE invalidate_deactivated_queries_cache();
