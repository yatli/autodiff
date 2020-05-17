// Q-Space 16-bit, 3 ext bit, 1 growth bit adder

module q16_add(
  input [15:0] input_a,
  input [15:0] input_b,
  output [15:0] output_z
);

wire [15:0] scaled_a, scaled_b;
wire [15:0] sum;
wire G;
wire OV;

q16_scaleup scaleup(
  .input_a(input_a),
  .input_b(input_b),
  .output_a(scaled_a),
  .output_b(scaled_b),
  .G(G)
);

i16_add adder(
  .a(scaled_a[14:0]),
  .b(scaled_b[14:0]),
  .cin(0),
  .sum(sum),
  .cout(OV)
);

assign output_z = (OV && !G) ? { 5'b10000, sum[14:4] } :  // shifted
                  (OV && G ) ? { 1'b1, sum[14:0] } : // XXX should be min/max saturation
                  { G, sum[14:0] };

endmodule
