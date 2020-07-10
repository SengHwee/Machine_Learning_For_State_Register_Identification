`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    23:25:07 09/20/2006 
// Design Name: 
// Module Name:    testbench 
// Project Name:                            
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments:                                             
//
//////////////////////////////////////////////////////////////////////////////////

module rom(addr, data, dataeno);

   input [10:0] addr;
   inout [7:0] data;
   input dataeno;

   reg [7:0] datao;

   always @(addr) case (addr)

      `include "test.rom" // get contents of memory
     
      default datao = 8'h76; // hlt
   
   endcase

   // Enable drive for data output
   assign data = dataeno ? datao: 8'bz;
   
endmodule
