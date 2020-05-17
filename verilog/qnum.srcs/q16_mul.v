// Q-Space 16-bit, 3ext bit, 1 growth bit multiplier

module q16_mul(
  input [15:0] input_a,
  input [15:0] input_b,
  output [15:0] output_z
);

wire [15:0] scaled_a, scaled_b;
wire [31:0] product;
wire G;

TopMultiplier multiplier(
  .x_in(scaled_a[14:0]),
  .y_in(scaled_b[14:0]),
  .result_out(product)
);

q16_scaleup scaleup(
  .input_a(input_a),
  .input_b(input_b),
  .output_a(scaled_a),
  .output_b(scaled_b),
  .G(G)
);

// full range: [31 .. 0]
// shift8:     [22 .. 8]
// shift11:    [25 .. 11]

wire [14:0] shift_product = G ? product[25:11] : 
                            product[22:8];

assign output_z = { G, shift_product }; // XXX wrong

endmodule
