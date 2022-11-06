# Bender's decomposition

Implementation of Bender's decomposition method for solving Mixed-Integer Linear 
Programs (MILPs) of the following form:


$$
\\begin{aligned}
\\min\_{x,y} \\quad & c\^Tx + f\^Ty  \\\\
\\textrm{s.t.} \\quad 
 & Ax + By \\ge b\\\\
 & Dy \\ge d \\\\
 & x \\ge 0  \\\\
 & y \\ge 0  \\\\
 & y \\text{ integer}
\\end{aligned}
$$