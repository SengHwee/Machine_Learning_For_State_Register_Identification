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

module selectone(addr, data, write, read, selectin, selectout, reset);

   input [15:0] addr;     // address to match, 6 bits
   inout [7:0] data;      // CPU data
   input       write;     // CPU write
   input       read;      // CPU read
   input       selectin;  // select for read/write
   output      selectout; // resulting select
   input       reset;     // reset

   reg  [7:0] mask;  // mask/control, 7:2 is mask, 1: I/O or /mem, 0: on/off
   reg  [7:2] comp;  // Compare value
   wire [5:0] iaddr; // multiplexed address
   reg  [7:0] datai; // data from output selector

   // select what part of address, upper or lower byte, we compare, based on
   // I/O or memory address
   assign iaddr = mask[1] ? addr[7:2]: addr[15:10];

   // Form select based on match
   assign selectout = ((iaddr & mask[7:2]) == comp) & mask[0];

   always @(addr, write, read, reset, selectin, data, comp, mask)
      if (reset) begin

      comp <= 6'b0; // clear registers
      mask <= 8'b0;

   end else if (write&selectin) begin

      if (addr[0]) comp <= data[7:2]; // write comparitor data
      else mask <= data; // write mask data

   end else begin

      if (addr[0]) datai <= {comp, 2'b0}; // read comparitor data
      else datai <= mask; // read mask data

   end

   assign data = read&selectin ? datai: 8'bz; // enable output data

endmodule
