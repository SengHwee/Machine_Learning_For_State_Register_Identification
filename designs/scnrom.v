`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:02:03 10/22/2006 
// Design Name: 
// Module Name:    vgachr 
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
// Simulation plugs exist in this code. Look for "????? SIMULATION PLUG"
//
// Debug plugs exist in this code. Look for "????? DEBUG PLUG"
//
//////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// TERMINAL EMULATOR
//
// Emulates an ADM 3A dumb terminal, with a MITS serial I/O board interface.
// Two ports are emulated:
//
// 0: Control
// 1: Data
//
// The MITS serial card has all of its configuration performed by jumpers on the
// card, which means there is no programming configuration required. The data
// to the terminal is sent out of the data port, while the data from the 
// keyboard, which is not yet implemented, is read from the same data port.
// The parity is ignored on output, and set to 0 on input.
//
// The control register ignores all writes, and returns $80 if the terminal is
// busy, otherwise $00. This is the output ready to send bit. The busy bit
// reflects if the state machine is processing an operation. This is a cheat
// that only works with our emulated terminal/serial board pair, because 
// normally there is no way for the local CPU to know that the remote terminal
// is busy. This can lead to problems in the real world, and it's why Unix
// "termcap" terminal descriptions commonly have waiting periods perscribed
// time consuming operations like screen clear. This emulated terminal is
// "ideal" in that it accounts for all of this automatically, but it does not
// hurt the realisim of the emulation. An application that performs delays based
// on real terminal operations won't be incorrect because it performs a delay,
// but a stupid application that relies on this "smart" implementation might
// fail to run on the real thing.
//
// The ADM 3A terminal emulation is based on "ADM-3A Operators manual" of 1979,
// and "ADM 3A Dumb Terminal Users Reference Manual" of April, 1986, and the
// Unix termcap definition of the terminal.
//
// Several actions of terminals are typically not listed in the documentation,
// and the ADM 3A is no different. That's why termcap exists. The following
// actions were derived from the termcap definition:
//
// 1. The screen clear command also homes the cursor.
// 2. Giving the terminal a right cursor command while at the right side of the
// 3. Screen wraps the cursor around to the right, one line down. If the cursor
//    is at the 80th collumn of the 24th line, it will then scroll.
// 4. Down cursor (line feed) on the 24th line scrolls the screen.
//
// Note that rule (3) causes the terminal to be unable to ever write a character
// to the collumn 80 character of line 24, which is interesting if you are
// writing a full screen editor. The ADM 3A had a switch for this behavior, but
// the user could hardly be expected to open the side panel and flip this switch
// to be able to edit.
//
// The following actions are unknown:
//
// 1. A cursor left command given when the cursor is at the left side of the 
//    screen has unknown effect. According to termcap, it is not to go to the
//    end of the next line up. I have guessed here that it is to simply
//    refuse to move.
// 2. A cursor up command given when the cursor is at the top of the screen
//    has unknown effect. I have guessed here that is is to simply refuse to
//    move.
//
// These are terminal features there are no plans to implement:
//
// 1. ENQ or answerback mode. This appears to require operator setup, the manual
// does not specify any useful default value for the 32 byte answerback, 
// otherwise it might serve as sort of an early plug and play. If there is an
// application out there that uses this, I would put it in.
// 2. Bell (where would it go?).
// 3. Ctrl-N and Ctrl-O keyboard locking and extention port. I'm not sure 
// locking has a good use, nor is all that great an idea.
// 4. Any of the setup modes or features, as operated by the keyboard.
//
// Not implemented in this version:
//
// 1. Reduced intensity mode.
// 2. Graphics mode.
//
// The ADM 3A terminal will pretty much serve as an upward compatible version
// of the ASR-33 teletypes (Western Union Surplus) that were also commonly used
// in the time of the Altair. The ADM 3A was a common replacement for such
// units.
//
// Bugs/problems/issues:
//
// 1. The emulation was occasionally observed to hang. The CPU was still running
// and polling the keyboard, but no reply is received when keys are hit. The rdy
// signal is not coming from the low level keyboard logic.
// 2. Cursor left at left hand side does not refuse to move (see above).
// 3. moving to the 80th collum on line 24 and typing a character does not cause
// the screen to scroll.
// 4. On sign-on, the first two characters are missing the top row. I suspect
// this is due to the start of frame character preloading that is done.
// 5. \ESCGc or set attributes does not work from the keyboard unless you hit
// caps lock or hold the shift key down before and after the escape. It should
// work anytime the upper case G is hit.
//

//
// Terminal height and width
//
`define scnchrs 80 // width
`define scnlins 24 // height

//
// Terminal states
//
`define term_idle    5'h00 // idle
`define term_wrtstd2 5'h01 // write standard character #2
`define term_wrtstd3 5'h02 // write standard character #3
`define term_wrtstd4 5'h03 // write standard character #4
`define term_clear   5'h04 // clear screen and home cursor
`define term_clear2  5'h05 // clear screen and home cursor #2
`define term_clear3  5'h06 // clear screen and home cursor #3
`define term_clear4  5'h07 // clear screen and home cursor #4
`define term_fndstr  5'h08 // find start of current line
`define term_scroll  5'h09 // scroll screen
`define term_scroll1 5'h0a // scroll screen #1
`define term_scroll2 5'h0b // scroll screen #2
`define term_scroll3 5'h0c // scroll screen #3
`define term_scroll4 5'h0d // scroll screen #4
`define term_scroll5 5'h0e // scroll screen #5
`define term_scroll6 5'h0f // scroll screen #6
`define term_esc     5'h10 // escape
`define term_poscur  5'h11 // position cursor
`define term_poscur2 5'h12 // position cursor #2
`define term_attset  5'h13 // set screen attributes

//
// Terminal attribute bits
//
`define attr_blank   5'b00001 // blank
`define attr_blink   5'b00010 // blink
`define attr_reverse 5'b00100 // reverse
`define attr_under   5'b01000 // underline
`define attr_rinten  5'b10000 // reduced intensity

module scnrom(addr, data);

   input  [7:0] addr;
   output [7:0] data;

   reg [7:0]  data;

   always @(addr) case (addr)

      8'h00: data = 8'h00; // 
      8'h01: data = 8'h00; // f9
      8'h02: data = 8'h00; // 
      8'h03: data = 8'h00; // f5
      8'h04: data = 8'h00; // f3
      8'h05: data = 8'h00; // f1
      8'h06: data = 8'h00; // f2
      8'h07: data = 8'h00; // f12
      8'h08: data = 8'h00; // 
      8'h09: data = 8'h00; // f10
      8'h0A: data = 8'h00; // f8
      8'h0B: data = 8'h00; // f6
      8'h0C: data = 8'h00; // f4
      8'h0D: data = 8'h09; // tab
      8'h0E: data = 8'h60; // `
      8'h0F: data = 8'h00; // 
      8'h10: data = 8'h00; // 
      8'h11: data = 8'h00; // lft alt
      8'h12: data = 8'h00; // lft shift
      8'h13: data = 8'h00; // 
      8'h14: data = 8'h00; // left ctl
      8'h15: data = 8'h71; // q
      8'h16: data = 8'h31; // 1
      8'h17: data = 8'h00; // 
      8'h18: data = 8'h00; // 
      8'h19: data = 8'h00; // 
      8'h1A: data = 8'h7a; // z
      8'h1B: data = 8'h73; // s
      8'h1C: data = 8'h61; // a
      8'h1D: data = 8'h77; // w
      8'h1E: data = 8'h32; // 2
      8'h1F: data = 8'h00; // 
      8'h20: data = 8'h00; // 
      8'h21: data = 8'h63; // c
      8'h22: data = 8'h78; // x
      8'h23: data = 8'h64; // d
      8'h24: data = 8'h65; // e
      8'h25: data = 8'h34; // 4
      8'h26: data = 8'h33; // 3
      8'h27: data = 8'h00; // 
      8'h28: data = 8'h00; // 
      8'h29: data = 8'h20; // sp
      8'h2A: data = 8'h76; // v
      8'h2B: data = 8'h66; // f
      8'h2C: data = 8'h74; // t
      8'h2D: data = 8'h72; // r
      8'h2E: data = 8'h35; // 5
      8'h2F: data = 8'h00; // 
      8'h30: data = 8'h00; // 
      8'h31: data = 8'h6e; // n
      8'h32: data = 8'h62; // b
      8'h33: data = 8'h68; // h
      8'h34: data = 8'h67; // g
      8'h35: data = 8'h79; // y
      8'h36: data = 8'h36; // 6
      8'h37: data = 8'h00; // 
      8'h38: data = 8'h00; // 
      8'h39: data = 8'h00; // 
      8'h3A: data = 8'h6d; // m
      8'h3B: data = 8'h6a; // j
      8'h3C: data = 8'h75; // u
      8'h3D: data = 8'h37; // 7
      8'h3E: data = 8'h38; // 8
      8'h3F: data = 8'h00; // 
      8'h40: data = 8'h00; // 
      8'h41: data = 8'h2c; // ,
      8'h42: data = 8'h6b; // k
      8'h43: data = 8'h69; // i
      8'h44: data = 8'h6f; // o
      8'h45: data = 8'h30; // 0
      8'h46: data = 8'h39; // 9
      8'h47: data = 8'h00; // 
      8'h48: data = 8'h00; // 
      8'h49: data = 8'h2e; // .
      8'h4A: data = 8'h2f; // /
      8'h4B: data = 8'h6c; // l
      8'h4C: data = 8'h3b; // ;
      8'h4D: data = 8'h70; // p
      8'h4E: data = 8'h2d; // -
      8'h4F: data = 8'h00; // 
      8'h50: data = 8'h00; // 
      8'h51: data = 8'h00; // 
      8'h52: data = 8'h27; // '
      8'h53: data = 8'h00; // 
      8'h54: data = 8'h5b; // [
      8'h55: data = 8'h3d; // =
      8'h56: data = 8'h00; // 
      8'h57: data = 8'h00; // 
      8'h58: data = 8'h00; // caps lock
      8'h59: data = 8'h00; // rgt shift
      8'h5A: data = 8'h0D; // ent
      8'h5B: data = 8'h5d; // ]
      8'h5C: data = 8'h00; // 
      8'h5D: data = 8'h5c; // \
      8'h5E: data = 8'h00; // 
      8'h5F: data = 8'h00; // 
      8'h60: data = 8'h00; // 
      8'h61: data = 8'h00; // 
      8'h62: data = 8'h00; // 
      8'h63: data = 8'h00; // 
      8'h64: data = 8'h00; // 
      8'h65: data = 8'h00; // 
      8'h66: data = 8'h08; // bcksp
      8'h67: data = 8'h00; // 
      8'h68: data = 8'h00; // 
      8'h69: data = 8'h31; // 1
      8'h6A: data = 8'h00; // 
      8'h6B: data = 8'h34; // 4
      8'h6C: data = 8'h37; // 7
      8'h6D: data = 8'h00; // 
      8'h6E: data = 8'h00; // 
      8'h6F: data = 8'h00; // 
      8'h70: data = 8'h30; // 0
      8'h71: data = 8'h2e; // .
      8'h72: data = 8'h32; // 2
      8'h73: data = 8'h35; // 5
      8'h74: data = 8'h36; // 6
      8'h75: data = 8'h38; // 8
      8'h76: data = 8'h1B; // esc
      8'h77: data = 8'h00; // num lock
      8'h78: data = 8'h00; // f11
      8'h79: data = 8'h2b; // +
      8'h7A: data = 8'h33; // 3
      8'h7B: data = 8'h2d; // -
      8'h7C: data = 8'h2a; // -
      8'h7D: data = 8'h39; // 9
      8'h7E: data = 8'h00; // scl lock
      8'h7F: data = 8'h00; // 
      8'h80: data = 8'h00; // 
      8'h81: data = 8'h00; // 
      8'h82: data = 8'h00; // 
      8'h83: data = 8'h00; // f7
      8'h84: data = 8'h00; // 
      8'h85: data = 8'h00; // 
      8'h86: data = 8'h00; // 
      8'h87: data = 8'h00; // 
      8'h88: data = 8'h00; // 
      8'h89: data = 8'h00; // 
      8'h8A: data = 8'h00; // 
      8'h8B: data = 8'h00; // 
      8'h8C: data = 8'h00; // 
      8'h8D: data = 8'h00; // 
      8'h8E: data = 8'h00; // 
      8'h8F: data = 8'h00; // 

      default data = 8'b00000000; // blank
   
   endcase
   
endmodule
