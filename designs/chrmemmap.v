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

module chrmemmap(rst_n, clk, r, g, b, hsync_n, vsync_n, addr, read, write, 
                 data, attr, cursor);

      input         rst_n;   // reset
      input         clk;     // master clock
      output [2:0]  r, g, b; // R,G,B color output buses
      output        hsync_n; // horizontal sync pulse
      output        vsync_n; // vertical sync pulse
      input  [10:0] addr;    // address to read or write
      input         read;    // read from address
      input         write;   // write from address
      inout  [7:0]  data;    // data to be written/read
      inout  [4:0]  attr;    // attributes to be written/read
      input [10:0]  cursor;  // cursor address

   reg [15:0] pixeldata; // 16 bit pixel feed

   reg  [6:0]  chrcnt;   // character counter
   reg  [4:0]  rowcnt;   // character row counter
   reg  [4:0]  lincnt;   // line counter
   reg  [10:0] scnadr;   // screen character buffer address
   reg  [7:0]  curchr;   // current character indexed by scnadr
   reg  [4:0]  curatr;   // current attribute indexed by scnadr
   reg  [1:0]  fchsta;   // character fetch state, 0 = load high, 1 = load low
   wire [10:0] chradr;   // character generator address
   wire [7:0]  chrdata;  // character generator data
   reg  [7:0]  datao;    // intermediate for data output
   reg  [4:0]  attro;    // intermediate for attribute output
   reg  [7:0]  pixdatl;  // pixel data low holding
   reg  [31:0] blinkcnt; // blink cycle counter
   reg         blon;     // blink on/off

   // storage for character based screen

   reg [7:0] scnbuf[1919:0]; // screen
   reg [4:0] atrbuf[1919:0]; // attributes

   assign rst = ~rst_n; // change reset polarity

   vga vgai(.rst(rst), .clk(clk), .pixel_data_in(pixeldata), .rd(rd), .eof(eof),
            .r(r), .g(g), .b(b), .hsync_n(hsync_n), .vsync_n(vsync_n), 
            .blank(blank));

   chrrom crom(chradr, chrdata); // place character generator

   // run the blink cycle counter
   always @(posedge clk)
      if (rst) begin

      blinkcnt <= 0; // clear blink cycle
      blon <= 0; // clear cursor blink on

   end else begin

      blinkcnt <= blinkcnt+1; // count blink cycle
      // check blink cycle maxed, recycle if so
      if (blinkcnt >= `blinkmax) begin

         blinkcnt <= 0; // clear blink count
         blon <= ~blon; // flip blink state

      end

   end

   // run the character to screen scan
   always @(posedge clk)
      if (rst) begin // if reset

      // ????? SIMULATION PLUG
      // starting the character counter at the end of the line allows the scan
      // to cross the area the CPU is filling, and is ok at hardware time.
      // chrcnt <= 7'h0; // clear counters
      chrcnt <= 80-20; // clear counters
      // ????? SIMULATION PLUG
      // Starting the row count at 1 allows pixels to appear on the simulation,
      // and produces only a single bad line in real hardware
      rowcnt <= 5'h0;
      // rowcnt <= 5'h1;
      lincnt <= 5'h0;
      scnadr <= 11'h0;
      fchsta <= 0;

   end else if (eof) begin // if end of frame

      chrcnt <= 7'h0; // clear counters
      rowcnt <= 5'h0;
      lincnt <= 5'h0;
      scnadr <= 11'h0;
      fchsta <= 1; // set to fetch first set of characters

   end else if (rd || fchsta) begin

      if (fchsta == 1 || fchsta == 3) begin

         // advance counters
         if (chrcnt < 79) chrcnt <= chrcnt+1; // next character count
         else begin // next row
      
            chrcnt <= 0; // reset character
            if (rowcnt < 19) rowcnt <= rowcnt+1; // next character row
            else begin // next line
      
               rowcnt <= 0; // reset row
               lincnt <= lincnt+1; // next line
               scnadr <= scnadr+80; // advance character fetch
      
            end
      
         end

      end

      // Choose high or low character, and next state. Note we have to flip the
      // characters left to right to be correct.
      case (fchsta)

         0: fchsta <= 1; // delay until rd cycle is over

         1: begin
   
            // Set low bits of pixel register, and reverse if cursor matches the
            // current position, or if reverse attribute is on. Turn it all off
            // if blank is on.
            if ((scnadr+chrcnt == cursor)^curatr[2])
               pixdatl <= ~(curatr[0]|(curatr[1]&blon)? 8'h00: // blink and blank
                             (curatr[3]&rowcnt == 14? 8'hff: // underline 
                              { chrdata[0], chrdata[1], chrdata[2], chrdata[3],
                                chrdata[4], chrdata[5], chrdata[6], chrdata[7] }));
            else 
               pixdatl <= curatr[0]|(curatr[1]&blon)? 8'h00: // blink and blank
                             (curatr[3]&rowcnt == 14? 8'hff: // underline 
                              { chrdata[0], chrdata[1], chrdata[2], chrdata[3],
                                chrdata[4], chrdata[5], chrdata[6], chrdata[7] });
            fchsta <= 2; // next state
   
         end

         2: fchsta <= 3; // delay a cycle for ROM read time

         3: begin
   
            // Set low bits of pixel register, and reverse if cursor matches the
            // current position, or if reverse attribute is on. Turn it all off
            // if blank is on.
            if ((scnadr+chrcnt == cursor)^curatr[2]) 
               pixeldata <= pixdatl |
                 (~(curatr[0]|(curatr[1]&blon)? 8'h00: // blink and blank
                    (curatr[3]&rowcnt == 14? 8'hff: // underline 
                     { chrdata[0], chrdata[1], chrdata[2], chrdata[3],
                       chrdata[4], chrdata[5], chrdata[6], chrdata[7] })) & 
                  8'hff) << 8;
            else 
               pixeldata <= pixdatl |
                 (curatr[0]|(curatr[1]&blon)? 8'h00: // blink and blank
                    (curatr[3]&rowcnt == 14? 8'hff: // underline 
                     { chrdata[0], chrdata[1], chrdata[2], chrdata[3],
                       chrdata[4], chrdata[5], chrdata[6], chrdata[7] })) << 8;
            fchsta <= 0; // back to start
            
         end
   
      endcase

   end

   // operate dual port screen character RAM
   always @(posedge clk) begin

      // set current indexed character without parity
      curchr <= scnbuf[scnadr+chrcnt] & 8'h7f;
      if (write) scnbuf[addr] <= data;
      datao <= scnbuf[addr];

   end

   // operate dual port screen attribute RAM
   always @(posedge clk) begin

      // set current indexed character without parity
      curatr <= atrbuf[scnadr+chrcnt];
      if (write) atrbuf[addr] <= attr;
      attro <= atrbuf[addr];

   end

   // create character address from character in buffer and current row
   assign chradr =
      (curchr < 8'h20 || curchr == 8'h7f) ? 11'h0: (curchr-8'h20)*20+rowcnt;

   // Enable drive for data output
   assign data = read ? datao: 8'bz;
   assign attr = read ? attro: 8'bz;

endmodule
