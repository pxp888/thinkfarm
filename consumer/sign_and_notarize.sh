#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DETECTED_APP=$(find "${SCRIPT_DIR}/dist" -maxdepth 1 -name "thinkfarm-client*.app" 2>/dev/null | head -n 1)
DEFAULT_APP=$(basename "${DETECTED_APP:-thinkfarm-client.app}")

APP_NAME="${1:-${DEFAULT_APP}}"
APP_PATH="${SCRIPT_DIR}/dist/${APP_NAME}"
ZIP_PATH="${SCRIPT_DIR}/dist/${APP_NAME%.app}.zip"
DMG_NAME="${APP_NAME%.app}-macOS.dmg"
DMG_PATH="${SCRIPT_DIR}/dist/${DMG_NAME}"
IDENTITY="Developer ID Application: paul xavier perrine (B4LQ2JCC67)"
ENTITLEMENTS="${SCRIPT_DIR}/entitlements.plist"

echo "=== 1. Signing all embedded binaries and frameworks inside ${APP_NAME} ==="
find "${APP_PATH}" -type f | while read -r f; do
    if file "$f" | grep -q "Mach-O"; then
        codesign --force --options runtime --timestamp --sign "${IDENTITY}" "$f"
    fi
done

find "${APP_PATH}" -name "*.framework" | sort -r | while read -r fw; do
    codesign --force --options runtime --timestamp --sign "${IDENTITY}" "$fw" || true
done

echo "=== 2. Signing main executable ==="
find "${APP_PATH}/Contents/MacOS" -type f | while read -r exec_file; do
    if file "$exec_file" | grep -q "Mach-O"; then
        codesign --force --options runtime --timestamp --entitlements "${ENTITLEMENTS}" --sign "${IDENTITY}" "$exec_file"
    fi
done

echo "=== 3. Deep signing app bundle ==="
codesign --force --options runtime --timestamp --entitlements "${ENTITLEMENTS}" --sign "${IDENTITY}" "${APP_PATH}"

echo "=== 4. Verifying local signature ==="
codesign -vvv --deep --strict "${APP_PATH}"

echo "=== 5. Building DMG Packaging for ${APP_NAME} ==="
STAGING_DIR="${SCRIPT_DIR}/dist/dmg_staging"
rm -rf "${STAGING_DIR}" "${DMG_PATH}"
mkdir -p "${STAGING_DIR}"

echo "Copying app to staging..."
cp -R "${APP_PATH}" "${STAGING_DIR}/"

echo "Creating Applications folder shortcut..."
ln -s /Applications "${STAGING_DIR}/Applications"

echo "Building DMG using hdiutil..."
hdiutil create -volname "${APP_NAME%.app}" -srcfolder "${STAGING_DIR}" -ov -format UDZO "${DMG_PATH}"

echo "Cleaning up staging directory..."
rm -rf "${STAGING_DIR}"

echo "=== 6. Signing DMG with Developer ID ==="
codesign --force --sign "${IDENTITY}" "${DMG_PATH}"

echo "Verifying DMG signature..."
codesign -vvv --strict "${DMG_PATH}"

echo "=== 7. Submitting DMG to Apple Notary Service ==="
xcrun notarytool submit "${DMG_PATH}" --keychain-profile "AC_PASSWORD" --wait

echo "=== 8. Stapling Notarization Ticket ==="
xcrun stapler staple "${DMG_PATH}"
xcrun stapler staple "${APP_PATH}"

echo "=== 9. Verifying Gatekeeper Acceptance ==="
spctl --assess --type execute --verbose "${APP_PATH}"
xcrun stapler validate "${DMG_PATH}"

echo ""
echo "=== Signing, DMG Creation, and Notarization Complete! ==="
echo "Final DMG Location: ${DMG_PATH}"

