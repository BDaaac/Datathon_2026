import asyncio

import anyio
from fastapi import HTTPException
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header


async def save_upload(request, path, settings):
    """Incremental multipart parsing: no full-body buffering or extra video copies."""
    kind, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise HTTPException(400, "Expected multipart/form-data with a boundary")
    limit = settings.max_upload_bytes
    filename, size, total, parts, ended = None, 0, 0, 0, False
    headers, field, value = {}, bytearray(), bytearray()
    with path.open("xb") as output:
        def part_begin():
            nonlocal parts
            parts += 1
            if parts > 1:
                raise HTTPException(422, "Exactly one file field is required")
            headers.clear()

        def header_field(data, start, end):
            field.extend(data[start:end])
            if len(field) > 256:
                raise HTTPException(400, "Multipart header too long")

        def header_value(data, start, end):
            value.extend(data[start:end])
            if len(value) > 8192:
                raise HTTPException(400, "Multipart header too long")

        def header_end():
            headers[bytes(field).lower()] = bytes(value)
            field.clear()
            value.clear()

        def headers_finished():
            nonlocal filename
            disposition, params = parse_options_header(headers.get(b"content-disposition", b""))
            if disposition != b"form-data" or params.get(b"name") != b"file" or b"filename" not in params:
                raise HTTPException(422, "Required multipart field: file")
            filename = params[b"filename"].decode("utf-8", errors="replace")

        def part_data(data, start, end):
            nonlocal size
            size += end - start
            if size > limit:
                raise HTTPException(413, "Video exceeds upload size limit")
            output.write(data[start:end])

        def finished():
            nonlocal ended
            ended = True

        parser = MultipartParser(boundary, {
            "on_part_begin": part_begin, "on_header_field": header_field, "on_header_value": header_value,
            "on_header_end": header_end, "on_headers_finished": headers_finished,
            "on_part_data": part_data, "on_end": finished,
        })
        try:
            async with asyncio.timeout(settings.upload_timeout):
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > limit + 65536:
                        raise HTTPException(413, "Request exceeds upload size limit")
                    await anyio.to_thread.run_sync(parser.write, chunk)
                parser.finalize()
        except TimeoutError:
            raise HTTPException(408, "Upload time limit exceeded") from None
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "Malformed or interrupted multipart upload") from None
        if filename is None or not ended:
            raise HTTPException(422, "Complete multipart field file is required")
    return filename, size
