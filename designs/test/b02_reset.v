module b02_reset(clock, RESET_G, nRESET_G, LINEA, U_REG);

input LINEA;
input RESET_G;
wire STATO_REG_0_; 
wire STATO_REG_1_; 
wire STATO_REG_2_; 
output U_REG;
wire _abc_181_new_n10_; 
wire _abc_181_new_n11_; 
wire _abc_181_new_n12_; 
wire _abc_181_new_n13_; 
wire _abc_181_new_n15_; 
wire _abc_181_new_n16_; 
wire _abc_181_new_n17_; 
wire _abc_181_new_n18_; 
wire _abc_181_new_n19_; 
wire _abc_181_new_n20_; 
wire _abc_181_new_n21_; 
wire _abc_181_new_n22_; 
wire _abc_181_new_n23_; 
wire _abc_181_new_n25_; 
wire _abc_181_new_n26_; 
wire _abc_181_new_n27_; 
wire _abc_181_new_n28_; 
wire _abc_181_new_n29_; 
wire _abc_181_new_n30_; 
wire _abc_181_new_n31_; 
wire _abc_181_new_n33_; 
wire _abc_181_new_n34_; 
wire _abc_181_new_n35_; 
wire _abc_181_new_n36_; 
wire _abc_181_new_n37_; 
wire _abc_181_new_n38_; 
input clock;
wire n10; 
wire n14; 
wire n19; 
wire n24; 
input nRESET_G;
AND2X2 AND2X2_1 ( .A(_abc_181_new_n10_), .B(STATO_REG_2_), .Y(_abc_181_new_n11_));
AND2X2 AND2X2_10 ( .A(_abc_181_new_n10_), .B(_abc_181_new_n25_), .Y(_abc_181_new_n33_));
AND2X2 AND2X2_11 ( .A(_abc_181_new_n34_), .B(STATO_REG_0_), .Y(_abc_181_new_n35_));
AND2X2 AND2X2_12 ( .A(_abc_181_new_n15_), .B(STATO_REG_1_), .Y(_abc_181_new_n36_));
AND2X2 AND2X2_13 ( .A(_abc_181_new_n36_), .B(_abc_181_new_n12_), .Y(_abc_181_new_n37_));
AND2X2 AND2X2_2 ( .A(_abc_181_new_n12_), .B(nRESET_G), .Y(_abc_181_new_n13_));
AND2X2 AND2X2_3 ( .A(_abc_181_new_n11_), .B(_abc_181_new_n13_), .Y(n10));
AND2X2 AND2X2_4 ( .A(_abc_181_new_n15_), .B(LINEA), .Y(_abc_181_new_n16_));
AND2X2 AND2X2_5 ( .A(_abc_181_new_n17_), .B(_abc_181_new_n10_), .Y(_abc_181_new_n18_));
AND2X2 AND2X2_6 ( .A(_abc_181_new_n21_), .B(_abc_181_new_n12_), .Y(_abc_181_new_n22_));
AND2X2 AND2X2_7 ( .A(_abc_181_new_n25_), .B(STATO_REG_2_), .Y(_abc_181_new_n26_));
AND2X2 AND2X2_8 ( .A(_abc_181_new_n20_), .B(STATO_REG_0_), .Y(_abc_181_new_n29_));
AND2X2 AND2X2_9 ( .A(_abc_181_new_n28_), .B(_abc_181_new_n30_), .Y(_abc_181_new_n31_));
DFFPOSX1 DFFPOSX1_1 ( .CLK(clock), .D(n24), .Q(STATO_REG_0_));
DFFPOSX1 DFFPOSX1_2 ( .CLK(clock), .D(n19), .Q(STATO_REG_1_));
DFFPOSX1 DFFPOSX1_3 ( .CLK(clock), .D(n10), .Q(U_REG));
DFFPOSX1 DFFPOSX1_4 ( .CLK(clock), .D(n14), .Q(STATO_REG_2_));
INVX1 INVX1_1 ( .A(STATO_REG_1_), .Y(_abc_181_new_n10_));
INVX1 INVX1_2 ( .A(STATO_REG_0_), .Y(_abc_181_new_n12_));
INVX1 INVX1_3 ( .A(STATO_REG_2_), .Y(_abc_181_new_n15_));
INVX1 INVX1_4 ( .A(nRESET_G), .Y(_abc_181_new_n19_));
INVX1 INVX1_5 ( .A(_abc_181_new_n20_), .Y(_abc_181_new_n21_));
INVX1 INVX1_6 ( .A(LINEA), .Y(_abc_181_new_n25_));
OR2X2 OR2X2_1 ( .A(_abc_181_new_n16_), .B(_abc_181_new_n12_), .Y(_abc_181_new_n17_));
OR2X2 OR2X2_10 ( .A(_abc_181_new_n37_), .B(_abc_181_new_n19_), .Y(_abc_181_new_n38_));
OR2X2 OR2X2_11 ( .A(_abc_181_new_n35_), .B(_abc_181_new_n38_), .Y(n19));
OR2X2 OR2X2_2 ( .A(STATO_REG_2_), .B(LINEA), .Y(_abc_181_new_n20_));
OR2X2 OR2X2_3 ( .A(_abc_181_new_n22_), .B(_abc_181_new_n19_), .Y(_abc_181_new_n23_));
OR2X2 OR2X2_4 ( .A(_abc_181_new_n23_), .B(_abc_181_new_n18_), .Y(n24));
OR2X2 OR2X2_5 ( .A(_abc_181_new_n16_), .B(STATO_REG_0_), .Y(_abc_181_new_n27_));
OR2X2 OR2X2_6 ( .A(_abc_181_new_n27_), .B(_abc_181_new_n26_), .Y(_abc_181_new_n28_));
OR2X2 OR2X2_7 ( .A(_abc_181_new_n29_), .B(STATO_REG_1_), .Y(_abc_181_new_n30_));
OR2X2 OR2X2_8 ( .A(_abc_181_new_n31_), .B(_abc_181_new_n19_), .Y(n14));
OR2X2 OR2X2_9 ( .A(_abc_181_new_n33_), .B(STATO_REG_2_), .Y(_abc_181_new_n34_));


endmodule