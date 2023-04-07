diff --git a/algos.yaml b/algos.yaml
index d30bd7a..db3fca5 100644
--- a/algos.yaml
+++ b/algos.yaml
@@ -542,6 +542,28 @@ float:
 
 
   euclidean:
+    countrymaam-kd:
+      docker-tag: ann-benchmarks-countrymaam
+      module: ann_benchmarks.algorithms.countrymaam
+      constructor: Countrymaam
+      base-args: ["@metric"]
+      run-groups:
+        kd:
+          arg-groups:
+            - {"index": ["kd_tree"], "trees": [8, 16],
+               "leafs":[8, 16]}
+          query-args: [[16, 32]]
+    countrymaam-rp:
+      docker-tag: ann-benchmarks-countrymaam
+      module: ann_benchmarks.algorithms.countrymaam
+      constructor: Countrymaam
+      base-args: ["@metric"]
+      run-groups:
+        rp:
+          arg-groups:
+            - {"index": ["rp_tree"], "trees": [8, 16],
+               "leafs":[8, 16]}
+          query-args: [[16, 32]]
     vamana(diskann):
       docker-tag: ann-benchmarks-diskann
       module: ann_benchmarks.algorithms.diskann
diff --git a/ann_benchmarks/runner.py b/ann_benchmarks/runner.py
index d2e896d..f4973b7 100644
--- a/ann_benchmarks/runner.py
+++ b/ann_benchmarks/runner.py
@@ -266,9 +266,11 @@ def run_docker(definition, dataset, count, runs, timeout, batch, cpu_limit,
 def _handle_container_return_value(return_value, container, logger):
     base_msg = 'Child process for container %s' % (container.short_id)
     if type(return_value) is dict: # The return value from container.wait changes from int to dict in docker 3.0.0
-        error_msg = return_value['Error']
         exit_code = return_value['StatusCode']
-        msg = base_msg + 'returned exit code %d with message %s' %(exit_code, error_msg)
+        msg = base_msg + 'returned exit code %d' % exit_code
+        if 'Error' in return_value:
+            error_msg = return_value['Error']
+            msg += 'with message %s' %(exit_code, error_msg)
     else: 
         exit_code = return_value
         msg = base_msg + 'returned exit code %d' % (exit_code)
