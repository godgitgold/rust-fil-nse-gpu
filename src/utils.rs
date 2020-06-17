use crate::GPUResult;
use ocl::Device;

pub fn get_bus_id(d: Device) -> GPUResult<u32> {
    const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
    let result = d.info_raw(CL_DEVICE_PCI_BUS_ID_NV)?;
    Ok((result[0] as u32)
        + ((result[1] as u32) << 8)
        + ((result[2] as u32) << 16)
        + ((result[3] as u32) << 24))
}
