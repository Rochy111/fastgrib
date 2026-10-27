/* Reference JPEG2000 codestreams through the system's libopenjp2, for
 * testing fastgrib's own codec.
 *   opj_tool enc IN.i32 OUT.j2k W H PREC SIGNED NRES CBW CBH MODE PROG TILE LAYERS IRREV PRC [RATE...]
 *   opj_tool dec IN.j2k OUT.i32          (prints W H PREC SIGNED)
 * IN.i32 / OUT.i32: little-endian int32 samples, row by row.
 * MODE: code-block style bits (1 bypass, 2 reset, 4 termall, 8 vsc, 16 pterm, 32 segsym)
 * PROG: 0 LRCP 1 RLCP 2 RPCL 3 PCRL 4 CPRL; TILE: tile size (0: one tile)
 * PRC: precinct size exponent for every level (0: default, the largest)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openjpeg.h>

static void quiet(const char *m, void *d) { (void)m; (void)d; }
static void loud(const char *m, void *d) { (void)d; fprintf(stderr, "opj: %s", m); }

static int enc(int argc, char **argv) {
    int w = atoi(argv[4]), h = atoi(argv[5]), prec = atoi(argv[6]), sgn = atoi(argv[7]);
    opj_cparameters_t p;
    opj_set_default_encoder_parameters(&p);
    p.numresolution = atoi(argv[8]);
    p.cblockw_init = atoi(argv[9]);
    p.cblockh_init = atoi(argv[10]);
    p.mode = atoi(argv[11]);
    p.prog_order = (OPJ_PROG_ORDER)atoi(argv[12]);
    int tile = atoi(argv[13]);
    if (tile > 0) { p.tile_size_on = OPJ_TRUE; p.cp_tdx = tile; p.cp_tdy = tile; }
    p.tcp_numlayers = atoi(argv[14]);
    p.irreversible = atoi(argv[15]);
    int prc = atoi(argv[16]);
    if (prc > 0) {
        p.csty |= 0x01;
        p.res_spec = p.numresolution;
        for (int r = 0; r < p.numresolution; r++) { p.prcw_init[r] = 1 << prc; p.prch_init[r] = 1 << prc; }
    }
    p.cp_disto_alloc = 1;
    for (int l = 0; l < p.tcp_numlayers; l++)
        p.tcp_rates[l] = (17 + l < argc) ? (float)atof(argv[17 + l]) : 0.0f;
    FILE *f = fopen(argv[2], "rb");
    int *buf = malloc(sizeof(int) * (size_t)w * h);
    if (fread(buf, sizeof(int), (size_t)w * h, f) != (size_t)w * h) return 2;
    fclose(f);
    opj_image_cmptparm_t c;
    memset(&c, 0, sizeof c);
    c.dx = c.dy = 1; c.w = w; c.h = h; c.prec = prec; c.sgnd = sgn;
    opj_image_t *im = opj_image_create(1, &c, OPJ_CLRSPC_GRAY);
    im->x0 = 0; im->y0 = 0; im->x1 = w; im->y1 = h;
    memcpy(im->comps[0].data, buf, sizeof(int) * (size_t)w * h);
    opj_codec_t *k = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(k, loud, NULL);
    opj_set_warning_handler(k, quiet, NULL);
    if (!opj_setup_encoder(k, &p, im)) return 3;
    opj_stream_t *s = opj_stream_create_default_file_stream(argv[3], OPJ_FALSE);
    if (!opj_start_compress(k, im, s) || !opj_encode(k, s) || !opj_end_compress(k, s)) return 4;
    opj_stream_destroy(s);
    opj_destroy_codec(k);
    opj_image_destroy(im);
    return 0;
}

static int dec(char **argv) {
    opj_dparameters_t p;
    opj_set_default_decoder_parameters(&p);
    opj_codec_t *k = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(k, loud, NULL);
    opj_set_warning_handler(k, quiet, NULL);
    opj_setup_decoder(k, &p);
    opj_stream_t *s = opj_stream_create_default_file_stream(argv[2], OPJ_TRUE);
    opj_image_t *im = NULL;
    if (!opj_read_header(s, k, &im) || !opj_decode(k, s, im) || !opj_end_decompress(k, s)) return 5;
    opj_image_comp_t *c = &im->comps[0];
    FILE *f = fopen(argv[3], "wb");
    fwrite(c->data, sizeof(int), (size_t)c->w * c->h, f);
    fclose(f);
    printf("%u %u %u %u\n", c->w, c->h, c->prec, c->sgnd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 17 && !strcmp(argv[1], "enc")) return enc(argc, argv);
    if (argc == 4 && !strcmp(argv[1], "dec")) return dec(argv);
    fprintf(stderr, "usage: see the source\n");
    return 1;
}
