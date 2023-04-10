diff --git a/plot.py b/plot.py
index f9784db..0ff628a 100644
--- a/plot.py
+++ b/plot.py
@@ -69,7 +69,7 @@ def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles,
     # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
     ax.legend(handles, labels, loc='center left',
               bbox_to_anchor=(1, 0.5), prop={'size': 9})
-    plt.grid(b=True, which='major', color='0.65', linestyle='-')
+    plt.grid(which='major', color='0.65', linestyle='-')
     plt.setp(ax.get_xminorticklabels(), visible=True)
 
     # Logit scale has to be a subset of (0,1)
