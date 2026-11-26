#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

struct scriptor_error {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

static void scriptor_error_exit(j_common_ptr cinfo) {
    struct scriptor_error *err = (struct scriptor_error *)cinfo->err;
    (*cinfo->err->format_message)(cinfo, err->message);
    longjmp(err->jump, 1);
}

static void scriptor_emit_message(j_common_ptr cinfo, int level) {
    (void)cinfo;
    (void)level;
}

/* Decodes a JPEG to interleaved RGB at 1/scale_denom size with the library defaults
 * (islow IDCT, fancy upsampling). Returns 0 on success with a malloc'd buffer in *out; otherwise a
 * nonzero status and a message in `message`. */
int scriptor_jpeg_decode_rgb(const unsigned char *data, size_t len, unsigned char **out,
                       unsigned int *width, unsigned int *height, unsigned int scale_denom,
                       char *message, size_t message_len) {
    struct jpeg_decompress_struct cinfo;
    struct scriptor_error err;
    unsigned char *buffer = NULL;
    *out = NULL;
    cinfo.err = jpeg_std_error(&err.pub);
    err.pub.error_exit = scriptor_error_exit;
    err.pub.emit_message = scriptor_emit_message;
    if (setjmp(err.jump)) {
        snprintf(message, message_len, "%s", err.message);
        jpeg_destroy_decompress(&cinfo);
        free(buffer);
        return 1;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, len);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    cinfo.scale_num = 1;
    cinfo.scale_denom = scale_denom ? scale_denom : 1;
    jpeg_start_decompress(&cinfo);
    *width = cinfo.output_width;
    *height = cinfo.output_height;
    size_t stride = (size_t)cinfo.output_width * 3;
    buffer = (unsigned char *)malloc(stride * cinfo.output_height);
    if (!buffer) {
        snprintf(message, message_len, "out of memory");
        jpeg_destroy_decompress(&cinfo);
        return 2;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row = buffer + (size_t)cinfo.output_scanline * stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    *out = buffer;
    return 0;
}

void scriptor_jpeg_free(unsigned char *buffer) { free(buffer); }
