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

module ram(addr, data, select, read, write, bootstrap, clock);

   input [9:0] addr;
   inout [7:0] data;
   input select;
   input read;
   input write;
   input clock;
   input bootstrap;

   reg [7:0] ramcore [1023:0]; // The ram store
   reg [7:0] datao;
   
   always @(negedge clock) 
      if (select) begin

         if (write) ramcore[addr] <= data;
         datao <= ramcore[addr];

      end

   // Enable drive for data output
   assign data = (select&read&~bootstrap) ? datao: 8'bz;
   
endmodule
